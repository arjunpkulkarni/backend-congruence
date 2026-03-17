# test_sophia_sessions.py
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

print("=== Testing Sophia Sessions ===\n")

# Check what key is loaded
supabase_key = os.getenv("SUPABASE_KEY")
print(f"SUPABASE_KEY loaded: {supabase_key[:50]}...")
print(f"Is it service_role? {'service_role' in supabase_key}\n")

# Create client
supabase = create_client(
    "https://bjwtvklzfewuebppjyso.supabase.co",
    supabase_key
)

sophia_id = "02801945-7212-4b63-8c2a-92586e94d851"

# Test the exact query from list_sessions
print("Testing session_videos query...")
try:
    response = supabase.table("session_videos")\
        .select("*, session_analysis(*)")\
        .eq("patient_id", sophia_id)\
        .order("created_at", desc=True)\
        .execute()
    
    print(f"✅ Query succeeded!")
    print(f"   Found {len(response.data)} videos\n")
    
    if response.data:
        for i, video in enumerate(response.data, 1):
            print(f"{i}. {video.get('title')}")
            print(f"   ID: {video.get('id')}")
            print(f"   Status: {video.get('status')}")
            print(f"   Duration: {video.get('duration_seconds')}s")
            analysis = video.get('session_analysis', [])
            if analysis:
                print(f"   Analysis: avg_tecs={analysis[0].get('avg_tecs')}")
            print()
    else:
        print("❌ No videos found for Sophia")
        print("\nTrying to list ALL videos to see what's there...")
        all_videos = supabase.table("session_videos").select("patient_id, title").limit(5).execute()
        print(f"Sample videos in database:")
        for v in all_videos.data:
            print(f"  - Patient: {v.get('patient_id')}, Title: {v.get('title')}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Also test using the data_access function
print("\n=== Testing data_access.list_sessions ===")
try:
    from app.services.data_access import list_sessions
    sessions = list_sessions(sophia_id)
    print(f"list_sessions returned: {len(sessions)} sessions")
    for s in sessions:
        print(f"  - {s.get('title')}")
except Exception as e:
    print(f"Error: {e}")