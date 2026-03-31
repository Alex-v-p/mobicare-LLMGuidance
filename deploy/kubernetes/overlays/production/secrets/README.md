kubectl create secret generic llmguidance-secrets \
  -n llmguidance \
  --from-env-file=deploy/kubernetes/overlays/production/secrets/secret.env




  kubectl create secret tls llmguidance-api-tls \
  -n llmguidance \
  --cert=deploy/kubernetes/overlays/production/secrets/tls/fullchain.pem \
  --key=deploy/kubernetes/overlays/production/secrets/tls/privkey.pem