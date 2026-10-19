"""
FastAPI Simulation Server & Multi-Stream WebSocket Telemetry Hub.
Provides:
1. Live Digital Twin Network Simulation (/ws/telemetry)
2. Live Interactive RL Training Studio (/ws/training)
3. Live Head-to-Head Arena (OSPF vs RL) (/ws/arena)
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent
from src.agents.live_trainer import LiveTrainingManager, LiveHeadToHeadArena
from src.sdn.mininet_exporter import MininetExporter

app = FastAPI(title="Green SDN RL Network Simulator & Training Studio", version="2.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SimulationEngine:
    def __init__(self):
        self.env = GreenNetworkEnv(max_steps_per_episode=1000)
        self.active_mode = "RL"
        self.manual_action = 7
        self.step_delay = 1.0
        self.clients = set()

        self.rl_agent = GreenRLAgent(env=self.env, algo="PPO")
        self.ospf_agent = OSPFBaselineAgent(env=self.env)
        self.heuristic_agent = StaticHeuristicAgent(env=self.env)

        self.last_obs, self.last_info = self.env.reset(seed=42)
        self.prev_active_cores = set(self.last_info.get("active_cores", []))

    def reload_agent(self):
        self.rl_agent = GreenRLAgent(env=self.env, algo="PPO")
        self.ospf_agent = OSPFBaselineAgent(env=self.env)
        self.heuristic_agent = StaticHeuristicAgent(env=self.env)

    def set_topology(self, topo_name: str):
        """Switches active simulation network topology (fattree_k4, fattree_k6, nsfnet)."""
        self.env = GreenNetworkEnv(topology_type=topo_name, max_steps_per_episode=1000)
        self.reload_agent()
        self.last_obs, self.last_info = self.env.reset(seed=42)
        self.prev_active_cores = set(self.last_info.get("active_cores", []))

    def set_scenario(self, scenario_name: str):
        """Sets active workload profile: diurnal, datacenter_pareto, flash_crowd."""
        self.env.traffic_gen.set_workload_profile(scenario_name)

    def set_mode(self, mode: str):
        if mode in ["RL", "OSPF", "HEURISTIC", "MANUAL"]:
            self.active_mode = mode

    def set_burst(self, enabled: bool):
        self.env.traffic_gen.set_burst_mode(enabled)

    def set_intensity(self, intensity: float):
        self.env.traffic_gen.set_intensity(intensity)

    def generate_ai_explanation(self, info: Dict[str, Any], action: int) -> Dict[str, Any]:
        import datetime
        active_cores = set(info.get("active_cores", []))
        prev_cores = getattr(self, "prev_active_cores", active_cores)
        self.prev_active_cores = active_cores

        demand = int(info.get("total_demand_mbps", 0))
        savings = info.get("energy_savings_percent", 0)
        delay = info.get("avg_delay_ms", 1.0)
        max_util = float(info.get("max_utilization", 0.0))
        util_pct = int(max_util * 100)
        burst = self.env.traffic_gen.burst_mode
        active_count = len(active_cores)
        total_cores = len(self.env.core_switches)

        newly_opened = active_cores - prev_cores
        newly_closed = prev_cores - active_cores

        readable_opened = ", ".join([c.replace("c_", "C") for c in sorted(newly_opened)])
        readable_closed = ", ".join([c.replace("c_", "C") for c in sorted(newly_closed)])
        readable_active = ", ".join([c.replace("c_", "C") for c in sorted(active_cores)])

        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        if self.active_mode == "OSPF":
            title = "OSPF Modu (Tüm Cihazlar Açık)"
            msg = f"Geleneksel yönlendirme devrede. Tüm {total_cores} omurga anahtarı açık tutuluyor. Enerji tasarrufu: %0 (İsraf yüksek)."
            type_class = "ospf"
        elif burst:
            title = "⚡ Ani Trafik Patlaması (Burst)"
            if active_count == total_cores:
                msg = f"Ağa ani trafik dalgası geldi ({demand} Mbps)! Tıkanıklık ve paket kaybını engellemek için anında tüm omurga anahtarlarını açtım ({active_count}/{total_cores} Core aktif: {readable_active}). Kesinti ve darboğaz önlendi."
            else:
                msg = f"Ağa ani trafik dalgası geldi ({demand} Mbps)! Tıkanıklığı dağıtmak için ek kapasiteye geçtim ({active_count}/{total_cores} Core aktif: {readable_active}). Veri akışı dengeleniyor."
            type_class = "burst"
        elif newly_opened:
            title = f"📈 Kapasite Artırıldı (+{readable_opened})"
            msg = f"Trafik {demand} Mbps seviyesine yükseldi (en yoğun hat %{util_pct} doldu). Gecikmeyi {delay:.2f} ms altında tutmak için {readable_opened} anahtarını uyandırdım. Şu an {active_count} switch devrede, %{savings} tasarruf devam ediyor."
            type_class = "scale_up"
        elif newly_closed:
            title = f"🌙 Enerji Tasarrufu (-{readable_closed})"
            msg = f"Trafik {demand} Mbps seviyesine geriledi. Boşuna elektrik yakmamak için {readable_closed} anahtarını uykuya aldım. Tasarruf oranımız %{savings}'a yükseldi!"
            type_class = "scale_down"
        else:
            if active_count <= 1:
                title = "🌿 Maksimum Yeşil Tasarruf"
                msg = f"Trafik sakin ({demand} Mbps). Yalnızca {readable_active} açık tutularak %{savings} enerji tasarrufu yapıyorum. Bağlantı kalitesi kusursuz."
                type_class = "eco"
            elif active_count < total_cores:
                title = "⛅ Dengeli Çalışma"
                msg = f"Trafik {demand} Mbps düzeyinde dengeli. Açık olan {active_count} switch ({readable_active}) veri akışını rahatça taşıyor. %{savings} enerji tasarrufu korunuyor."
                type_class = "balanced"
            else:
                title = "⚡ Pik Yük Koruması"
                msg = f"Yoğun trafik ({demand} Mbps) güvenle taşınıyor. Tüm {total_cores} omurga anahtarı devrede, sıfır paket kaybı hedefleniyor."
                type_class = "peak"

        return {
            "title": title,
            "message": msg,
            "type": type_class,
            "time": now_str,
            "active_cores": list(active_cores),
            "demand_mbps": demand,
            "savings_pct": savings
        }

    def trigger_step(self) -> Dict[str, Any]:
        if self.active_mode == "RL":
            action = self.rl_agent.predict(self.last_obs, deterministic=True)
        elif self.active_mode == "OSPF":
            action = self.ospf_agent.predict(self.last_obs)
        elif self.active_mode == "HEURISTIC":
            action = self.heuristic_agent.predict(self.last_obs)
        else:
            action = self.manual_action

        self.last_obs, reward, terminated, truncated, self.last_info = self.env.step(action)
        if terminated or truncated:
            self.last_obs, self.last_info = self.env.reset()

        ai_speech = self.generate_ai_explanation(self.last_info, action)

        payload = dict(self.last_info)
        payload["mode"] = self.active_mode
        payload["reward"] = round(reward, 2)
        payload["burst_mode"] = self.env.traffic_gen.burst_mode
        payload["traffic_intensity"] = getattr(self.env.traffic_gen, "traffic_intensity", 1.0)
        payload["ai_speech"] = ai_speech
        payload["nodes"] = [
            {
                "id": node,
                "type": self.env.topology.graph.nodes[node].get("type", "switch"),
                "x": self.env.topology.graph.nodes[node].get("x", 0),
                "y": self.env.topology.graph.nodes[node].get("y", 0),
                "is_active": self.env.switch_states.get(node, True)
            }
            for node in self.env.topology.graph.nodes
        ]
        payload["links"] = [
            {
                "source": u,
                "target": v,
                "load": self.last_info["link_loads"].get(f"{u}->{v}", 0.0),
                "utilization": self.last_info["link_utilizations"].get(f"{u}->{v}", 0.0),
                "is_active": (self.env.switch_states.get(u, True) and self.env.switch_states.get(v, True))
            }
            for (u, v) in self.env.links
        ]
        payload["active_demands"] = [
            {
                "flow_id": d.flow_id,
                "src": d.src,
                "dst": d.dst,
                "demand_mbps": d.demand_mbps,
                "is_burst": d.is_burst
            }
            for d in self.env.active_demands
        ]
        return payload


sim_engine = SimulationEngine()
training_manager = LiveTrainingManager()
arena = LiveHeadToHeadArena()
if os.path.exists("models/ppo_green_agent.zip"):
    arena.load_rl_agent("models/ppo_green_agent.zip")


@app.get("/")
def get_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Dashboard is loading...</h2>")


@app.get("/report")
def get_project_report():
    report_path = os.path.join("results", "PROJECT_REPORT.html")
    if os.path.exists(report_path):
        return FileResponse(report_path)
    return HTMLResponse("<h2>Rapor henuz olusturulmadi. Lutfen 'python scripts/generate_report.py' calistirin.</h2>")


@app.get("/results/{filename}")
def get_results_file(filename: str):
    file_path = os.path.join("results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("File not found", status_code=404)


@app.post("/api/topology/{name}")
def set_topology(name: str):
    sim_engine.set_topology(name)
    return {"status": "ok", "topology": name}


@app.post("/api/scenario/{name}")
def set_scenario(name: str):
    sim_engine.set_scenario(name)
    return {"status": "ok", "scenario": name}


@app.post("/api/mode/{mode}")
def set_simulation_mode(mode: str):
    sim_engine.set_mode(mode)
    return {"status": "ok", "mode": sim_engine.active_mode}


@app.post("/api/burst/{state}")
def toggle_burst(state: str):
    enabled = (state.lower() in ["true", "1", "on"])
    sim_engine.set_burst(enabled)
    return {"status": "ok", "burst_mode": enabled}


@app.post("/api/traffic-intensity/{intensity}")
def set_traffic_intensity(intensity: float):
    sim_engine.set_intensity(intensity)
    return {"status": "ok", "intensity": intensity}


@app.post("/api/export-mininet")
def export_mininet_files():
    MininetExporter.export_all()
    return {"status": "ok", "message": "Mininet and Ryu scripts generated in /mininet_scripts"}


class TrainRequest(BaseModel):
    algo: str = "PPO"
    timesteps: int = 5000
    lr: float = 0.0003
    topology: Optional[str] = None


@app.post("/api/train/start")
def start_live_training(req: TrainRequest):
    topo = req.topology or getattr(sim_engine.env, "topology_type", "fattree_k4")
    res = training_manager.start_training(algo=req.algo, timesteps=req.timesteps, lr=req.lr, topology_type=topo)
    return res


@app.post("/api/train/stop")
def stop_live_training():
    res = training_manager.stop_training()
    return res


@app.post("/api/train/apply")
def apply_trained_model():
    try:
        sim_engine.reload_agent()
        topo = getattr(sim_engine.env, "topology_type", "fattree_k4")
        arena.load_rl_agent()
        algo = sim_engine.rl_agent.algo_name if sim_engine.rl_agent else "RL"
        return {"status": "ok", "message": f"Eğitilen {algo} modeli ({topo}) aktif simülatöre ve test arenasına başarıyla yüklendi!"}
    except Exception as e:
        return {"status": "error", "message": f"Model yüklenirken hata oluştu: {str(e)}"}


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    sim_engine.clients.add(websocket)
    try:
        while True:
            data = sim_engine.trigger_step()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(sim_engine.step_delay)
    except WebSocketDisconnect:
        sim_engine.clients.discard(websocket)
    except Exception:
        sim_engine.clients.discard(websocket)


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Poll training queue
            while not training_manager.telemetry_queue.empty():
                item = training_manager.telemetry_queue.get_nowait()
                await websocket.send_text(json.dumps(item))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/arena")
async def websocket_arena(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = arena.step()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def start_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
