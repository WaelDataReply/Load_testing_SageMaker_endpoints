import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from transformers import AutoTokenizer
sys.path.append("utils")
from get_metrics import get_metrics_from_cloudwatch

import sys
import time
import concurrent.futures
from tqdm import tqdm
import json
 

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")
    
# Hub Model configuration. https://huggingface.co/models
hub = {
    'HF_MODEL_ID':'mistralai/Mistral-7B-Instruct-v0.1',
#	'HF_MODEL_ID':'meta-llama/Meta-Llama-3-8B',
	'SM_NUM_GPUS': json.dumps(1),
	'HUGGING_FACE_HUB_TOKEN': '<your-token>'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	image_uri=get_huggingface_llm_image_uri("huggingface",version="1.4.2"),
	env=hub,
	role=role, 
)

# deploy model to SageMaker Inference
llm = huggingface_model.deploy(
	initial_instance_count=1,
	instance_type="ml.g5.48xlarge",
	container_startup_health_check_timeout=300,
  )

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
 
# Generation arguments
parameters = {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.9,
    "max_new_tokens": 250,
    "return_full_text": False,
}
 
# The function to perform a single request
def make_request(payload):
    try:
        llm.predict(
            data={
                "inputs": tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": payload
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "parameters": parameters,
            }
        )
        return 200
    except Exception as e:
        print(e)
        return 500
    
# Main function to run the load test
def run_load_test(total_requests, concurrent_users):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # Prepare a list of the same inputs to hit multiple times
        tasks = ["Write a long story about llamas and why should protect them."] * total_requests
        start_time = time.time()
        
        # run the requests
        results = list(tqdm(executor.map(make_request, tasks), total=total_requests, desc="Running load test"))
        end_time = time.time()
        
        print(f"Total time for {total_requests} requests with {concurrent_users} concurrent users: {end_time - start_time:.2f} seconds")
        print(f"Successful rate: {results.count(200) / total_requests * 100:.2f}%")
        # Get the metrics
        metrics = get_metrics_from_cloudwatch(
            endpoint_name=llm.endpoint_name,
            st=int(start_time),
            et=int(end_time),
            cu=concurrent_users,
            total_requests=total_requests,
            boto3_session=sess.boto_session
        )
        # store results
        with open("results.json", "w") as f:
            json.dump(metrics, f)
        # print results
        # print(f"Llama 3 8B results on `g5.2xlarge`:")
        print(f"Mistral 7B Instruct results on `g5.48xlarge`:")
        print(f"Throughput: {metrics['Thorughput (tokens/second)']:,.2f} tokens/s")
        print(f"Latency p(50): {metrics['Latency (ms/token) p(50)']:,.2f} ms/token")
        return metrics

# Run the load test
concurrent_users = 5
number_of_requests = 100
res = run_load_test(number_of_requests, concurrent_users)