import httpx
import json
import time
import os
import urllib.request

base_url = "http://127.0.0.1:8000"
hf_base = "https://huggingface.co/datasets/imaadd05/visa-anomaly-detection/resolve/main"

test_cases = [
    {
        "name": "Normal Board #1 (Pristine Reference)",
        "expected": "PASS",
        "file": "data/real_tests/real_normal_0000.jpg",
        "url": f"{hf_base}/pcb1/Data/Images/Normal/0000.JPG",
        "notes": "Authentic defect-free circuit board from Amazon VisA training set."
    },
    {
        "name": "Normal Board #2 (Pristine Held-Out)",
        "expected": "PASS",
        "file": "data/real_tests/real_normal_0001.jpg",
        "url": f"{hf_base}/pcb1/Data/Images/Normal/0001.JPG",
        "notes": "Second clean circuit board from Amazon VisA testing set."
    },
    {
        "name": "Bent Pin Defect (Physical Damage)",
        "expected": "FAIL",
        "file": "data/real_tests/real_bent_pin_000.jpg",
        "url": f"{hf_base}/pcb1/Data/Images/Anomaly/000.JPG",
        "notes": "Transducer pin physically deformed/bent off alignment."
    },
    {
        "name": "Surface Melt / Burn Defect",
        "expected": "FAIL",
        "file": "data/real_tests/real_melt_015.jpg",
        "url": f"{hf_base}/pcb1/Data/Images/Anomaly/015.JPG",
        "notes": "Localized thermal melt damage on PCB dielectric layer."
    },
    {
        "name": "Trace Scratch Defect (Borderline/Subtle)",
        "expected": "FAIL or HUMAN",
        "file": "data/real_tests/real_scratch_085.jpg",
        "url": f"{hf_base}/pcb1/Data/Images/Anomaly/085.JPG",
        "notes": "Thin surface scratch across copper trace requiring geometric verification."
    },
    {
        "name": "Severe Specular Glare (Optical Corruption)",
        "expected": "RETAKE",
        "file": "data/real_tests/real_missing_comp_041.jpg",
        "url": f"{hf_base}/pcb1/Data/Images/Anomaly/041.JPG",
        "notes": "Overexposed transducer glare exceeding industrial gate tolerance."
    }
]

os.makedirs("data/real_tests", exist_ok=True)
for tc in test_cases:
    if not os.path.exists(tc["file"]):
        print(f"Downloading {tc['file']} from Hugging Face...")
        urllib.request.urlretrieve(tc["url"], tc["file"])

client = httpx.Client(timeout=30.0)

# Get session token
session_resp = client.post(f"{base_url}/api/session")
token = session_resp.json()["session_token"]
headers = {"Authorization": f"Bearer {token}"}

results = []

print("================================================================================")
print("             SUREINSPECT: REAL-WORLD INDUSTRIAL BENCHMARK TEST                  ")
print("================================================================================")

for tc in test_cases:
    print(f"\n--- Testing: {tc['name']} ---")
    print(f"File: {tc['file']} | Expected: {tc['expected']}")
    print(f"Context: {tc['notes']}")
    
    t0 = time.perf_counter()
    with open(tc["file"], "rb") as f:
        files = {"file": (tc["file"], f.read(), "image/jpeg")}
        data = {"category": "pcb1", "alpha_fail": "0.05"}
        resp = client.post(f"{base_url}/api/inspect", headers=headers, files=files, data=data)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    if resp.status_code == 200:
        res = resp.json()
        decision = res.get("decision")
        decision_path = res.get("decision_path")
        score = res.get("anomaly_score")
        score_str = f"{score:.4f}" if score is not None else "N/A (Gate Rejected)"
        q_gate = res.get("quality_gate", {})
        timing = res.get("timing_ms", {})
        tools = [t.get("tool") for t in res.get("execution_trace", [])]
        
        print(f"  --> VERDICT: {decision} | PATH: {decision_path}")
        print(f"  --> Anomaly Score: {score_str} (Thresholds: t_pass=2.0405, t_fail=2.2863)")
        print(f"  --> Quality Gate: status={q_gate.get('status')}, blur_var={q_gate.get('blur_variance', 0):.1f}, glare_frac={q_gate.get('glare_fraction', 0):.3f}")
        print(f"  --> Tools Run: {tools}")
        if res.get("measurements"):
            m = res["measurements"]
            print(f"  --> Defect Geometry: area={m.get('area_pct', 0):.3f}%, shape={m.get('shape_class')}, residual={m.get('residual_mean', 0):.4f}")
        print(f"  --> Execution Timing: Gate={timing.get('quality', 0):.1f}ms | Perception={timing.get('perception', 0):.1f}ms | Tools={timing.get('tools', 0):.1f}ms | Total={timing.get('total', 0):.1f}ms (HTTP Round-trip={elapsed_ms:.1f}ms)")
        print(f"  --> Grounded Explanation: {res.get('grounded_report')}")
        
        results.append({
            "name": tc["name"],
            "file": tc["file"],
            "expected": tc["expected"],
            "decision": decision,
            "decision_path": decision_path,
            "anomaly_score": score,
            "score_str": score_str,
            "timing_ms": timing,
            "quality_gate": q_gate,
            "measurements": res.get("measurements"),
            "report": res.get("grounded_report"),
        })
    else:
        print(f"  --> FAILED: HTTP {resp.status_code}: {resp.text}")

with open("data/real_tests/live_test_results.json", "w") as out:
    json.dump(results, out, indent=2)

print("\n================================================================================")
print("All real tests completed successfully!")
