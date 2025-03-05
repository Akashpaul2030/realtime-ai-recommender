# Real-time AI Recommender

A production-ready system for real-time AI integration with e-commerce product data. This project provides vector similarity search, real-time product recommendations, and efficient data processing with low latency.

![Architecture Diagram](https://github.com/Akashpaul2030/realtime-ai-recommender/raw/main/docs/architecture.png)

## ✨ Key Features

- **Real-time Processing**: Product updates are processed instantly through Redis Streams
- **Vector Similarity Search**: Find similar products using efficient vector embeddings
- **Low Latency Recommendations**: Generate product recommendations with minimal time complexity
- **Scalable Architecture**: All components designed for horizontal scaling
- **Robust Error Handling**: Comprehensive error handling and logging
- **Production Ready**: Includes metrics, monitoring, and proper documentation

## 🏗️ Architecture

The system is built with a microservices-oriented architecture with the following components:

1. **API Layer**: FastAPI-based RESTful API for product management and recommendations
2. **Stream Processing**: Redis Streams for event-driven real-time data processing 
3. **Vector Store**: Redis with vector similarity search capability
4. **Embedding Generation**: TF-IDF-based text-to-vector conversion
5. **Recommendation Engine**: Real-time product recommendation generation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Redis 6.2+ with RedisJSON and RediSearch modules
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Akashpaul2030/realtime-ai-recommender.git
   cd realtime-ai-recommender
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Redis:
   ```bash
   # If using Docker:
   docker run -p 6379:6379 -d redislabs/redismod
   ```

### Running the Application

1. Start the stream consumer (in a separate terminal):
   ```bash
   python -m services.stream_consumer --consumer-id worker1
   ```

2. Start the API server:
   ```bash
   python -m api.app
   ```

3. Access the API documentation:
   ```
   http://localhost:8000/docs
   ```

## 📚 API Endpoints

### Products

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/products/` | POST | Create a new product |
| `/products/{product_id}` | GET | Get a product by ID |
| `/products/{product_id}` | PUT | Update an existing product |
| `/products/{product_id}` | DELETE | Delete a product |
| `/products/similar/{product_id}` | GET | Find similar products |
| `/products/search/text` | GET | Search products by text |

### Recommendations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations/{product_id}/similar` | GET | Get similar product recommendations |
| `/recommendations/category/{category}` | GET | Get popular products in a category |
| `/recommendations/personalized` | GET | Get personalized recommendations |
| `/recommendations/track-view` | POST | Track product views |
| `/recommendations/search` | GET | Search recommendations by text |

## 📊 Example Usage

### Creating a Product

```bash
curl -X POST "http://localhost:8000/products/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Premium Wireless Headphones",
    "description": "High-quality wireless headphones with noise cancellation, 30-hour battery life, and premium sound quality.",
    "category": "Electronics",
    "price": 249.99,
    "sku": "AUDIO-WH100",
    "attributes": {
      "color": "Black",
      "connectivity": "Bluetooth 5.0",
      "batteryLife": "30 hours"
    },
    "id": "headphones-001"
  }'
```

### Finding Similar Products

```bash
curl -X GET "http://localhost:8000/products/similar/headphones-001"
```

### Searching Products

```bash
curl -X GET "http://localhost:8000/products/search/text?query=wireless+audio"
```

## 📁 Project Structure

```
ecommerce_realtime_ai/
├── README.md
├── requirements.txt
├── config.py                 # Configuration settings
├── models/                   # AI models
│   ├── embeddings.py         # Text embedding generation
│   ├── similarity.py         # Similarity search
│   └── recommendations.py    # Recommendation algorithms
├── data/                     # Data schemas
│   ├── schemas.py            # Pydantic models
│   └── validation.py         # Data validation
├── services/                 # Core services
│   ├── stream_consumer.py    # Redis Streams consumer
│   ├── stream_producer.py    # Redis Streams producer
│   └── vector_store.py       # Vector similarity store
├── api/                      # API layer
│   ├── app.py                # FastAPI application
│   ├── routes/               # API endpoints
│   └── middleware/           # API middleware
└── utils/                    # Utilities
    ├── logging.py            # Logging configuration
    └── metrics.py            # Metrics collection
```

## 🔧 Performance Considerations

- The system uses lightweight models and efficient vector operations for low time complexity
- Redis vector search provides near real-time similarity calculations
- Stream processing ensures new products are immediately available for recommendations
- The architecture separates read and write paths for better scaling

## 🔄 Extending the System

- Add user behavior tracking for more personalized recommendations
- Implement A/B testing for recommendation algorithms
- Add batch processing for historical data
- Integrate with external e-commerce platforms

## 🔒 Security Considerations

- Implement proper authentication and authorization
- Add rate limiting to protect against abuse
- Ensure data validation on all inputs
- Consider encryption for sensitive data

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request