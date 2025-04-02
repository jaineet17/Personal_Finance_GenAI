#!/bin/bash

API_ENDPOINT="https://y5eg4wztz7.execute-api.us-east-1.amazonaws.com/dev"

function print_usage() {
  echo "Finance RAG API Management Script"
  echo "--------------------------------"
  echo "Usage: ./manage_api.sh [command]"
  echo ""
  echo "Commands:"
  echo "  health            - Check API health"
  echo "  query \"[query]\"   - Send a query to the API"
  echo "  logs              - View recent Lambda logs"
  echo "  redeploy          - Redeploy the API"
  echo "  remove            - Remove the deployment"
  echo ""
}

case "$1" in
  health)
    echo "Checking API health..."
    curl -s $API_ENDPOINT/health
    echo ""
    ;;
  query)
    if [ -z "$2" ]; then
      echo "Error: Missing query parameter"
      echo "Usage: ./manage_api.sh query \"your query here\""
      exit 1
    fi
    echo "Sending query: $2"
    curl -s -X POST $API_ENDPOINT/query \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"$2\"}"
    echo ""
    ;;
  logs)
    echo "Fetching recent logs..."
    aws logs get-log-events \
      --log-group-name '/aws/lambda/finance-rag-api-dev-api' \
      --log-stream-name $(aws logs describe-log-streams \
                         --log-group-name '/aws/lambda/finance-rag-api-dev-api' \
                         --order-by LastEventTime \
                         --descending \
                         --limit 1 \
                         --query 'logStreams[0].logStreamName' \
                         --output text) \
      --limit 20
    ;;
  redeploy)
    echo "Redeploying API..."
    ./deploy_aws.sh
    ;;
  remove)
    echo "Removing deployment..."
    serverless remove
    ;;
  *)
    print_usage
    ;;
esac 