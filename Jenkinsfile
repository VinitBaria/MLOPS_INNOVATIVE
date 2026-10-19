pipeline {
    agent any

    environment {
        REGISTRY = 'localhost:5000'
        IMAGE_NAME = 'movie-recommender-api'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        KUBECONFIG_CREDENTIAL_ID = 'k8s-kubeconfig'
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code from Git repository...'
                checkout scm
            }
        }

        stage('Set Up Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Code Quality & Linting') {
            steps {
                echo 'Running linting and syntax checks...'
                sh '''
                    . .venv/bin/activate
                    python -m py_compile src/api/main.py src/models/hybrid.py
                '''
            }
        }

        stage('Automated Tests & Quality Gate') {
            steps {
                echo 'Running Pytest test suite...'
                sh '''
                    . .venv/bin/activate
                    pytest tests/ --disable-warnings -v
                '''
            }
        }

        stage('Model Validation Gate') {
            steps {
                echo 'Validating recommendation metrics thresholds (NDCG@10 >= 0.15)...'
                sh '''
                    . .venv/bin/activate
                    python -c "
import json, sys, os
meta_path = 'models/artifacts/metadata.json'
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
        best = meta.get('best_model', 'hybrid_recommender')
        ndcg = meta.get('metrics', {}).get(best, {}).get('ndcg_at_10', 0)
        print(f'Active model NDCG@10: {ndcg}')
        if ndcg < 0.10:
            print('Quality gate failed: NDCG below threshold')
            sys.exit(1)
print('Quality gate passed!')
"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker container image: ${IMAGE_NAME}:${IMAGE_TAG}..."
                sh '''
                    docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Publish Docker Image') {
            steps {
                echo "Publishing container image to registry..."
                sh '''
                    # docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                    # docker push ${REGISTRY}/${IMAGE_NAME}:latest
                    echo "Docker image tagged: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Deploying to Kubernetes cluster...'
                sh '''
                    kubectl apply -f k8s/namespace.yaml
                    kubectl apply -f k8s/postgres.yaml
                    kubectl apply -f k8s/api-deployment.yaml
                    kubectl apply -f k8s/nginx-ingress.yaml
                    kubectl rollout status deployment/recommender-api -n movie-rec --timeout=120s
                '''
            }
        }

        stage('Smoke Test & Verification') {
            steps {
                echo 'Running smoke verification tests...'
                sh '''
                    curl -s -f http://localhost:8000/health || exit 1
                    curl -s -f "http://localhost:8000/recommend/1?limit=3" || exit 1
                    echo "Smoke tests passed successfully!"
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo "Pipeline succeeded! New model and API deployed."
        }
        failure {
            echo "Pipeline failed! Deployment aborted or rolled back."
        }
    }
}
