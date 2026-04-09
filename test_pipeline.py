"""
Quick smoke test for the background job pipeline.
Run with: python test_pipeline.py

Requires the server to be running: uvicorn app.main:app --port 8000
"""
import requests
import time
import sys
import json

BASE = "http://localhost:8000"

# Use a short public-domain audio clip for testing
# This is a 10-second WAV clip from the LibriSpeech dataset
TEST_AUDIO_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav"


def test_health():
    print("1. Testing /health ...")
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print(f"   OK: {r.json()}")


def test_process_session():
    print("\n2. Testing POST /process_session (should return instantly) ...")
    payload = {
        "audio_url": TEST_AUDIO_URL,
        "patient_id": "test-patient-001",
        "fast_mode": True,
        "no_facial_analysis": True,
        "skip_video_analysis": True,
        "cleanup_files": False,
    }
    start = time.time()
    r = requests.post(f"{BASE}/process_session", json=payload)
    elapsed = time.time() - start
    print(f"   Response time: {elapsed:.2f}s")
    assert r.status_code == 200, f"POST failed: {r.status_code} {r.text}"
    data = r.json()
    print(f"   OK: job_id={data['job_id']}, status={data['status']}")
    assert "job_id" in data, "Missing job_id in response"
    assert elapsed < 5, f"POST took {elapsed:.1f}s — should be instant"
    return data["job_id"]


def test_poll_job(job_id: str, timeout: int = 180):
    print(f"\n3. Polling GET /jobs/{job_id} until completion (timeout={timeout}s) ...")
    start = time.time()
    last_stage = ""
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/jobs/{job_id}")
        assert r.status_code == 200, f"GET failed: {r.status_code} {r.text}"
        data = r.json()

        stage = data.get("stage", "")
        status = data.get("status", "")
        progress = data.get("progress", 0)
        msg = data.get("message", "")

        if stage != last_stage:
            print(f"   [{time.time()-start:6.1f}s] stage={stage} progress={progress}% status={status} | {msg}")
            last_stage = stage

        if status == "completed":
            elapsed = time.time() - start
            print(f"\n   COMPLETED in {elapsed:.1f}s")
            result = data.get("result")
            if result:
                print(f"   - transcript: {len(result.get('transcript_text') or '')} chars")
                print(f"   - segments: {len(result.get('transcript_segments') or [])}")
                print(f"   - has notes: {result.get('notes') is not None}")
                notes = result.get("notes")
                if notes:
                    print(f"   - notes keys: {list(notes.keys())}")
                    print(f"   - has soap_note: {'soap_note' in notes}")
                    print(f"   - has transcript_summary: {'transcript_summary' in notes}")
                print(f"   - timeline entries: {len(result.get('timeline_json') or [])}")
                print(f"   - session_summary keys: {list((result.get('session_summary') or {}).keys())}")
            return data

        if status == "failed":
            print(f"\n   FAILED: {data.get('error') or data.get('message')}")
            return data

        time.sleep(3)

    print(f"\n   TIMEOUT after {timeout}s — last status: {status}")
    return None


def main():
    print("=" * 60)
    print("CONGRUENCE PIPELINE SMOKE TEST")
    print("=" * 60)

    test_health()
    job_id = test_process_session()
    result = test_poll_job(job_id)

    print("\n" + "=" * 60)
    if result and result.get("status") == "completed":
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED — check server logs and debug log")
    print("=" * 60)


if __name__ == "__main__":
    main()
