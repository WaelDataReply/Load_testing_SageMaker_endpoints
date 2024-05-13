# Load_testing_SageMaker_endpoints
## Description:
This repository provides a comprehensive guide and codebase for deploying a Large Language Model (LLM) from Hugging Face on AWS Sagemaker. It includes scripts and instructions for setting up the environment, deploying the model, and conducting load testing to evaluate the performance of the deployed endpoint.

## Table of Contents:
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Setup](#3-setup)
4. [Deployment](#4-deployment)
5. [Load Testing](#5-load-testing)
6. [Conclusion](#6-conclusion)
7. [Contributing](#7-contributing)
8. [License](#8-license)

## 1. Introduction:
This repository aims to simplify the process of deploying LLMs from Hugging Face on AWS Sagemaker and conducting load testing to assess the endpoint's performance. By following the steps outlined here, users can quickly set up their environment, deploy a model, and evaluate its scalability and response times under different loads.

## 2. Prerequisites:
Before proceeding, ensure you have the following prerequisites installed and configured:
- AWS CLI
- Python 3.x
- Docker
- Hugging Face Transformers library
- Jupyter Notebook (optional, for testing the deployed endpoint)

## 3. Setup:
Clone this repository to your local machine:
```bash
git clone https://github.com/WaelDataReply/Load_testing_SageMaker_endpoints.git
cd Load_testing_SageMaker_endpoints
```
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```
Configure your AWS credentials using AWS CLI:
```bash
aws configure
```

## 4. Deployment:
Follow the steps provided in the `deploy_llm_and_load_testing_notebook.ipynb` file to deploy the LLM on AWS Sagemaker using the Hugging Face Inference API.

## 5. Load Testing:
After deploying the model, run load tests to evaluate its performance under various traffic conditions. Refer to the instructions in the `deploy_llm_and_load_testing_notebook.ipynb` or `deploy_llms_and_load_testing.py` file for conducting load testing

## 6. Conclusion:
This repository simplifies the process of deploying LLMs from Hugging Face on AWS Sagemaker and assessing their performance through load testing. By following the provided guides and scripts, users can efficiently deploy models and ensure their scalability and reliability in production environments.

## 7. Contributing:
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## 8. License:
This project is licensed under the MIT License. See the `LICENSE` file for details.

For any questions or support, feel free to contact [Wael SAIDENI](w.saideni@reply.com).

Happy deploying and testing!
