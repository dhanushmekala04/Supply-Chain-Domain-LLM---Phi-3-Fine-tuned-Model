
# Supply Chain Domain LLM - Phi-3 Fine-tuned Model

A specialized language model fine-tuned on supply chain graph data for generating Neo4j Cypher queries from natural language.

---

## Overview

This model converts natural language questions about supply chain operations into executable Cypher queries for Neo4j graph databases. It's trained on 10,000+ examples based on the Kaggle Supply Chain dataset structure.

---

## Model Information

- **Base Model:** microsoft/Phi-3-mini-4k-instruct
- **Fine-tuning Method:** QLoRA (4-bit quantization)
- **Training Data:** 10,000 supply chain query examples
- **Model Format:** GGUF (Q8_0 quantization)
- **Model Size:** ~4GB
- **Hugging Face:** dhanushmekaka/phi-3-mini-text-to-cypher-supply

---

## Graph Schema

The model is trained on the following Neo4j graph structure:

### Nodes

| Node Type | Description |
|-----------|-------------|
| **Product** | Items in the supply chain (SKU, name, price, etc.) |
| **Supplier** | Vendors who supply products |
| **Warehouse** | Storage locations for products |
| **CustomerOrder** | Customer purchase orders |
| **Shipment** | Delivery shipments |
| **TransportMode** | Transportation methods (truck, air, sea, rail) |
| **Route** | Shipping routes between locations |

### Relationships

| Relationship | From → To | Description |
|--------------|-----------|-------------|
| **SUPPLIES** | Supplier → Product | Supplier provides product |
| **STORED_AT** | Product → Warehouse | Product stored in warehouse |
| **ORDERS** | CustomerOrder → Product | Order contains product |
| **SHIPPED_FROM** | Shipment → Warehouse | Shipment originates from warehouse |
| **TO_ORDER** | Shipment → CustomerOrder | Shipment fulfills order |
| **USES_TRANSPORT** | Shipment → TransportMode | Shipment uses transport method |
| **FOLLOWS_ROUTE** | Shipment → Route | Shipment follows specific route |
| **CONNECTS** | Route → Warehouse | Route connects warehouses |

### Sample Graph Structure

```cypher
// Products and Suppliers
(s:Supplier {name: "ABC Corp", location: "California"})
-[:SUPPLIES]->
(p:Product {sku: "PROD-001", name: "Widget", price: 50.00})
-[:STORED_AT]->
(w:Warehouse {name: "West Coast Hub", location: "Los Angeles"})

// Orders and Shipments
(o:CustomerOrder {order_id: "ORD-123", date: "2024-01-15"})
-[:ORDERS]->
(p:Product)

(sh:Shipment {shipment_id: "SHIP-456", dispatch_date: "2024-01-16"})
-[:SHIPPED_FROM]->(w:Warehouse)
-[:TO_ORDER]->(o:CustomerOrder)
-[:USES_TRANSPORT]->(t:TransportMode {type: "Truck"})
-[:FOLLOWS_ROUTE]->(r:Route {name: "LA-SF", distance: 382})
```

---

## Installation & Setup

### Download Model

```bash
# Download from Hugging Face
wget https://huggingface.co/dhanushmekaka/phi-3-mini-text-to-cypher-supply


### Install Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows/Mac
# Download from https://ollama.ai
```

### Create Modelfile

Create a file named `Modelfile`:

```dockerfile
FROM ./phi3_gguf_q8_0.gguf

SYSTEM "<|system|>You are a helpful AI assistant trained on supply chain query generation.<|end|>"

TEMPLATE """<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|end|>"
PARAMETER num_ctx 2048
```

### Import Model

```bash
ollama create phi3-local -f Modelfile
```

---

## Usage

### Command Line

```bash
ollama run phi3-local "Top 5 products by total ordered quantity"
```

**Output:**
```cypher
MATCH (p:Product)<-[:ORDERS]-(o:CustomerOrder)
RETURN p.name, p.sku, SUM(o.quantity) AS total_quantity
ORDER BY total_quantity DESC
LIMIT 5
```

---

### Python API

```python
from ollama import Client

client = Client()

def generate_cypher(query):
    prompt = f"""<|user|>
{query}
<|end|>
<|assistant|>
"""
    
    response = client.chat(
        model="phi3-local",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

# Example usage
query = "Top 5 products by total ordered quantity"
cypher = generate_cypher(query)
print(cypher)
```

---

## Example Queries

### 1. Product Analysis

**Question:** "Show me products with low stock levels"

**Generated Cypher:**
```cypher
MATCH (p:Product)-[:STORED_AT]->(w:Warehouse)
WHERE p.stock_quantity < 100
RETURN p.sku, p.name, p.stock_quantity, w.name AS warehouse
ORDER BY p.stock_quantity ASC
```

---

### 2. Supplier Performance

**Question:** "Which suppliers provide the most products?"

**Generated Cypher:**
```cypher
MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)
RETURN s.name, s.location, COUNT(p) AS product_count
ORDER BY product_count DESC
```

---

### 3. Order Analytics

**Question:** "Find orders placed in the last 30 days"

**Generated Cypher:**
```cypher
MATCH (o:CustomerOrder)
WHERE o.date >= date() - duration('P30D')
RETURN o.order_id, o.date, o.total_amount
ORDER BY o.date DESC
```

---

### 4. Shipping Optimization

**Question:** "Show shipments using air transport with high costs"

**Generated Cypher:**
```cypher
MATCH (sh:Shipment)-[:USES_TRANSPORT]->(t:TransportMode)
WHERE t.type = 'Air' AND sh.cost > 1000
RETURN sh.shipment_id, sh.cost, sh.dispatch_date, sh.delivery_date
ORDER BY sh.cost DESC
```

---

### 5. Route Analysis

**Question:** "Find the longest delivery routes"

**Generated Cypher:**
```cypher
MATCH (r:Route)-[:CONNECTS]->(w:Warehouse)
RETURN r.name, r.distance, r.origin, r.destination
ORDER BY r.distance DESC
LIMIT 10
```

---

### 6. Warehouse Inventory

**Question:** "Which warehouses store products from supplier XYZ?"

**Generated Cypher:**
```cypher
MATCH (s:Supplier {name: 'XYZ'})-[:SUPPLIES]->(p:Product)-[:STORED_AT]->(w:Warehouse)
RETURN DISTINCT w.name, w.location, COUNT(p) AS product_count
ORDER BY product_count DESC
```

---

### 7. Lead Time Analysis

**Question:** "Calculate average lead time by transport mode"

**Generated Cypher:**
```cypher
MATCH (sh:Shipment)-[:USES_TRANSPORT]->(t:TransportMode)
RETURN t.type, 
       AVG(duration.between(sh.dispatch_date, sh.delivery_date).days) AS avg_lead_time_days
ORDER BY avg_lead_time_days ASC
```

---

### 8. Complex Multi-hop Query

**Question:** "Show products ordered by customers, their suppliers, and shipping costs"

**Generated Cypher:**
```cypher
MATCH (o:CustomerOrder)-[:ORDERS]->(p:Product)<-[:SUPPLIES]-(s:Supplier)
MATCH (sh:Shipment)-[:TO_ORDER]->(o)
RETURN p.name, p.sku, s.name AS supplier, 
       o.order_id, sh.cost AS shipping_cost,
       o.date AS order_date
ORDER BY sh.cost DESC
LIMIT 20
```

---

## Model Performance

### Training Metrics

- **Training Loss:** 0.42
- **Evaluation Loss:** 0.38
- **Training Time:** ~45 minutes (T4 GPU)
- **Dataset Size:** 10,000 examples

### Inference Performance

- **Latency:** 1-2 seconds per query (CPU)
- **Throughput:** 30-60 queries/minute
- **Memory Usage:** ~4GB RAM

---

## Supported Query Types

✅ **Aggregation Queries** - COUNT, SUM, AVG, MAX, MIN  
✅ **Filtering** - WHERE clauses with multiple conditions  
✅ **Sorting** - ORDER BY with ASC/DESC  
✅ **Limiting** - LIMIT and SKIP  
✅ **Date Filtering** - Recent orders, date ranges  
✅ **Multi-hop Relationships** - Complex graph traversals  
✅ **Pattern Matching** - Multiple MATCH clauses  
✅ **Grouping** - GROUP BY operations  

---

## Dataset Information

### Source
Based on **Kaggle Supply Chain Dataset** with 10,000 generated query pairs.

### Data Fields
- Product: SKU, name, price, category, stock_quantity
- Supplier: name, location, contact
- Warehouse: name, location, capacity
- CustomerOrder: order_id, date, total_amount, quantity
- Shipment: shipment_id, dispatch_date, delivery_date, cost
- TransportMode: type (Truck, Air, Sea, Rail)
- Route: name, distance, origin, destination

### Training Split
- **Training:** 8,500 examples (85%)
- **Validation:** 1,500 examples (15%)

---

## Integration with Neo4j

### Python Example

```python
from ollama import Client
from neo4j import GraphDatabase

class SupplyChainQueryBot:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.client = Client()
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
    
    def query(self, natural_language_query):
        # Generate Cypher from natural language
        prompt = f"""<|user|>
{natural_language_query}
<|end|>
<|assistant|>
"""
        
        response = self.client.chat(
            model="phi3-local",
            messages=[{"role": "user", "content": prompt}]
        )
        
        cypher = response["message"]["content"]
        print(f"Generated Cypher: {cypher}\n")
        
        # Execute on Neo4j
        with self.driver.session() as session:
            result = session.run(cypher)
            return result.data()
    
    def close(self):
        self.driver.close()

# Usage
bot = SupplyChainQueryBot(
    "bolt://localhost:7687",
    "neo4j",
    "password"
)

results = bot.query("Show top 5 suppliers by product count")
for record in results:
    print(record)

bot.close()
```


---
Project Structure
project-root/
├── conversion/
│   ├── convert-to-gguf.ipynb      # Model conversion notebook
│   └── coversion-read.md          # Conversion documentation
├── myenv/                          # Python virtual environment
├── Ollama/
│   ├── desktop.ini
│   ├── inference.png              # Model inference screenshot
│   ├── Modelfile                  # Ollama configuration
│   ├── ollama                     # Ollama executable
│   ├── ollama-read.md            # Ollama setup guide
│   └── phi3_gguf_q8_0.gguf       # Model weights (~4GB)
├── training/
│   ├── data/                      # Training dataset
│   ├── phi3_qlora/               # QLoRA training artifacts
│   ├── train_phi3_qlora.ipynb    # Training notebook
│   └── training-read.md          # Training documentation
├── main.py       # Local inference script
└── read.md                        # Main documentation

---

## License

- **Base Model:** MIT License (Microsoft Phi-3)
- **Fine-tuned Model:** [Your License]
- **Dataset:** Kaggle Supply Chain Dataset

---



## Resources

- **Model Card:** https://huggingface.co/dhanushmekaka/phi-3-mini-text-to-cypher-supply
- **Base Model:** https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **Neo4j Documentation:** https://neo4j.com/docs/
- **Ollama:** https://ollama.ai

