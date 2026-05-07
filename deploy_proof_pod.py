import requests
import time
import json

API_KEY = "YOUR_RUNPOD_API_KEY"
GRAPHQL_URL = "https://api.runpod.io/graphql"

def run_query(query, variables=None):
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    response = requests.post(GRAPHQL_URL, json={'query': query, 'variables': variables}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed with code {response.status_code}. {response.text}")

def get_gpu_types():
    query = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        communityPrice
        securePrice
      }
    }
    """
    return run_query(query)['data']['gpuTypes']

def deploy_pod(gpu_type_id):
    query = """
    mutation ($input: PodInput!) {
      podFindAndDeploy(input: $input) {
        id
        imageName
        runtime {
          status
        }
      }
    }
    """
    variables = {
        "input": {
            "gpuTypeId": gpu_type_id,
            "gpuCount": 1,
            "imageName": "runpod/parameter-golf:latest",
            "dockerArgs": "",
            "containerDiskInGb": 50,
            "volumeInGb": 50,
            "networkVolumeId": None,
            "name": "Parameter-Golf-Proof-Run"
        }
    }
    return run_query(query, variables)['data']['podFindAndDeploy']

def get_pod_status(pod_id):
    query = """
    query ($podId: String!) {
      pod(input: {podId: $podId}) {
          runtime {
            status
            ports {
              ip
              isPublic
              privatePort
              publicPort
            }
          }
      }
    }
    """
    variables = {"podId": pod_id}
    return run_query(query, variables)['data']['pod']

def main():
    print("Checking available H100s...")
    gpus = get_gpu_types()
    h100 = next((g for g in gpus if "H100" in g['displayName']), None)
    
    if not h100:
        print("Error: No H100 instance type found in API.")
        return

    print(f"Found {h100['displayName']} (ID: {h100['id']})")
    
    print("Deploying Pod...")
    pod = deploy_pod(h100['id'])
    pod_id = pod['id']
    print(f"Pod deployed! ID: {pod_id}")
    
    print("Waiting for pod to start (this usually takes 1-3 minutes)...")
    while True:
        status_data = get_pod_status(pod_id)
        if not status_data:
            print("Waiting for API to recognize Pod...")
            time.sleep(10)
            continue
            
        status = status_data['runtime']['status']
        print(f"Current Status: {status}")
        
        if status == "RUNNING":
            ports = status_data['runtime']['ports']
            ssh_port = next((p for p in ports if p['privatePort'] == 22), None)
            if ssh_port:
                print("\n" + "="*40)
                print("POD IS READY!")
                print(f"SSH command: ssh root@{ssh_port['ip']} -p {ssh_port['publicPort']} -i ~/.ssh/id_rsa")
                print("="*40)
                break
        
        if status in ["STOPPED", "EXITED"]:
            print(f"Error: Pod stopped unexpectedly with status {status}")
            break
            
        time.sleep(15)

if __name__ == "__main__":
    main()
