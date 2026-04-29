# Step F: Delete Mock Services via MCP

**Owner:** Sergeant (autonomous)
**Time:** 0 operator time
**Purpose:** Clear mock fleet before deploying real workers

---

## F1. MCP Script to Delete Mock Services

This script uses the Railway MCP connector to identify and delete all mock-only services.

```python
# Delete all mock services from Railway fleet
# Executed via MCP (railway_service_delete)

import json

def delete_mock_services():
    """
    Delete all services with TRAINER_KIND=mock.
    Uses tri_railway_mcp connector.
    """
    # Get current fleet status
    fleet = fleet_health()  # Returns dict with 'services' list

    deleted_count = 0
    for service in fleet.get("services", []):
        # Check if this is a mock service
        env_vars = service.get("environment", {})
        if env_vars.get("TRAINER_KIND") == "mock":
            service_id = service.get("id")
            service_name = service.get("name", service_id)

            print(f"Deleting mock service: {service_name} (id: {service_id})")

            # Delete via MCP
            result = railway_service_delete(
                service_id=service_id,
                confirm=True  # Required safety check
            )

            if result.get("success"):
                deleted_count += 1
                print(f"✓ Deleted: {service_name}")
            else:
                print(f"✗ Failed to delete: {service_name}")
                print(f"  Error: {result.get('error')}")

    print(f"\nDeleted {deleted_count} mock services")
    return deleted_count

# Execute
if __name__ == "__main__":
    delete_mock_services()
```

---

## F2. Verification After Deletion

After running the delete script:

```python
# Verify no mock services remain
fleet = fleet_health()

mock_count = sum(
    1 for s in fleet.get("services", [])
    if s.get("environment", {}).get("TRAINER_KIND") == "mock"
)

print(f"Remaining mock services: {mock_count}")
assert mock_count == 0, f"Expected 0 mock services, found {mock_count}"
```

---

## F3. Safety Checklist

- [ ] `confirm=True` is required for each deletion (cannot delete accidentally)
- [ ] Each deletion is logged to experience append (R7 mandatory)
- [ ] Fleet has at least 4 account slots available for real workers
- [ ] Operator notified of mock cleanup (via experience log)

---

**⏭️ When complete, proceed to Step G**
