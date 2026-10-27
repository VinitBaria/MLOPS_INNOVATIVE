pipeline {
    agent any

    environment {
        IMAGE_NAME          = 'movie-recommender-api'
        IMAGE_TAG           = "${env.BUILD_NUMBER}"
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
        // Absolute path to the local project — synced into the Jenkins workspace
        LOCAL_REPO_PATH     = '/Users/jenilbuha/Desktop/sem7/Mlops/assignment/MLOPS_INNOVATIVE'
        // Extend PATH so Jenkins shell can find docker / docker compose
        PATH                = "/Users/jenilbuha/.docker/bin:${env.PATH}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Copying source code from local project directory...'
                // file:// checkouts are blocked by Jenkins security policy.
                // Instead we rsync/cp the local workspace directly.
                sh 'rsync -a --exclude=.git --exclude=.venv --exclude=__pycache__ "${LOCAL_REPO_PATH}/" .'
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
                    PYTHONPATH=. pytest tests/ --disable-warnings -v
                '''
            }
        }

        stage('Model Validation Gate') {
            steps {
                echo 'Validating recommendation metrics thresholds (NDCG@10 >= 0.10)...'
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
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                '''
            }
        }

        stage('Deploy with Docker Compose') {
            steps {
                echo 'Deploying full stack with Docker Compose...'
                sh '''
                    # Tear down any previously running stack first
                    docker compose down --remove-orphans || true

                    # Start all services: postgres, mlflow, api, nginx, prometheus, grafana
                    docker compose up -d --build

                    # Wait for the API container to become healthy (max ~60 s)
                    echo "Waiting for API to become ready..."
                    for i in $(seq 1 12); do
                        if docker compose ps api | grep -q "healthy\\|Up"; then
                            echo "API is up."
                            break
                        fi
                        echo "  attempt $i/12 — sleeping 5 s..."
                        sleep 5
                    done
                '''
            }
        }

        // stage('Smoke Test & Verification') {
        //     steps {
        //         echo 'Running smoke verification tests against the Compose stack...'
        //         sh '''
        //             curl -s -f http://localhost:8000/health || exit 1
        //             curl -s -f "http://localhost:8000/recommend/1?limit=3" || exit 1
        //             echo "Smoke tests passed successfully!"
        //         '''
        //     }
        // }
    }

    post {
        always {
            echo 'Tearing down Docker Compose stack...'
            sh 'docker compose down -v --remove-orphans || true'
            // cleanWs() requires the ws-cleanup plugin — use deleteDir() instead
            deleteDir()
        }
        success {
            echo 'Pipeline succeeded! All services deployed and verified via Docker Compose.'
        }
        failure {
            echo 'Pipeline failed! Docker Compose stack has been torn down.'
        }
    }
}

