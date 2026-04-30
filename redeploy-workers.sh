#!/bin/bash
set -e

source .env

echo "========================================"
echo "IGLA Fleet Redeploy Script"
echo "========================================"
echo ""

# Service IDs from workers table
redeploy_service() {
  local SVC_NAME="$1"
  local SVC_ID="$2"

  # Extract account from service name
  local ACC=$(echo "$SVC_NAME" | sed 's/trios-train-v2-acc//; s/-s1597//')

  # Get token and env ID
  local TOKEN_VAR="RAILWAY_TOKEN_$ACC"
  local ENV_VAR="RAILWAY_ENVIRONMENT_ID_$ACC"

  local TOKEN="${!TOKEN_VAR}"
  local EID="${!ENV_VAR}"

  echo "----------------------------------------"
  echo "Account: $ACC"
  echo "Service: $SVC_NAME"
  echo "Service ID: $SVC_ID"
  echo "Environment ID: $EID"
  echo "----------------------------------------"

  # Redeploy
  echo "Redeploying..."
  local DEPLOY_RESULT=$(curl -s -X POST https://backboard.railway.app/graphql/v2 \
    -H "Project-Access-Token: $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { serviceInstanceRedeploy(serviceId: \\\"$SVC_ID\\\", environmentId: \\\"$EID\\\") }\"}")

  # Check result
  if echo "$DEPLOY_RESULT" | grep -q "errors"; then
    echo "❌ ERROR in redeploy:"
    echo "$DEPLOY_RESULT" | python3 -m json.tool 2>&1 || echo "$DEPLOY_RESULT"
  else
    echo "✅ Redeploy initiated"
    echo "$DEPLOY_RESULT" | python3 -m json.tool 2>&1 || echo "Raw: $DEPLOY_RESULT"
  fi
  echo ""
}

# Redeploy each worker
redeploy_service "trios-train-v2-acc0-s1597" "99d2a6a6-f3f2-42aa-9894-fdaafd8422ac"
redeploy_service "trios-train-v2-acc1-s1597" "ed44c56a-3bac-4815-bd74-51ee49c95747"
redeploy_service "trios-train-v2-acc2-s1597" "982361d5-ad80-4ba5-874a-06795e0cdda0"
redeploy_service "trios-train-v2-acc3-s1597" "982361d5-ad80-4ba5-874a-06795e0cdda0"
redeploy_service "trios-train-v2-acc4-s1597" "4db62ce6-6aa3-475d-b6c9-59756ca01605"
redeploy_service "trios-train-v2-acc5-s1597" "dd5de85b-bc49-432d-8e08-7b32f5874dbc"

echo "========================================"
echo "Redeploy script completed!"
echo "Workers should come online in 2-5 minutes"
echo "========================================"
