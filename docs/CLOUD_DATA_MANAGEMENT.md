# Finance RAG: Cloud Data Management with Free Tier Services

## Overview

This document outlines data management strategies for deploying the Finance RAG system using exclusively free tier cloud services. The focus is on optimizing data storage, retrieval, and processing while staying within free tier limitations.

## Free Tier Database Options

### MongoDB Atlas Free Tier
- **Limitations**:
  - 512MB storage
  - Shared RAM and CPU
  - No dedicated backup
  - Limited connections
- **Optimization Strategies**:
  - Sparse indexing
  - Data compression
  - Time-to-live collections
  - Capped collections for logs

### AWS DynamoDB Free Tier
- **Limitations**:
  - 25GB storage
  - 25 WCUs and 25 RCUs
  - No auto-scaling on free tier
- **Optimization Strategies**:
  - Single-table design
  - Sparse attributes
  - Optimized query patterns
  - Item size minimization

### SQLite with AWS S3
- **Configuration**:
  - SQLite database files stored in S3
  - Downloaded to Lambda /tmp space when needed
  - Periodic snapshots for durability
  - Lambda execution role permissions

## Data Storage Architecture

### Transaction Data
- **Primary Storage**: MongoDB Atlas (free tier)
- **Collection Structure**:
  - Transactions collection (limited to recent transactions)
  - Categories collection (reference data)
  - Metadata collection (system data)
- **Optimizations**:
  - JSON compression
  - Field selection (store only necessary fields)
  - Scheduled archiving to stay under 512MB

### Vector Embeddings
- **Storage Options**:
  - MongoDB Atlas Vector Search (limited in free tier)
  - AWS S3 + custom indexing
  - DynamoDB with binary attribute
- **Recommended Approach**:
  - Hybrid storage: recent vectors in MongoDB, historical in S3
  - Batch processing for vector operations
  - Caching frequently accessed vectors

### User Data
- **Storage**: DynamoDB (free tier)
- **Data Model**:
  - User profiles (minimal data)
  - User preferences
  - Query history (limited retention)
- **Security**:
  - Encrypted at rest (free with AWS managed keys)
  - Attribute-level encryption for sensitive data

## Data Lifecycle Management

### Data Retention Policy
- **Transaction Data**:
  - Keep last 3 months in primary storage
  - Archive older data to S3 (compressed)
  - Automatic pruning to stay within free tier limits
- **Vector Embeddings**:
  - Recent and frequently accessed in primary storage
  - Others archived to S3
  - Regenerate on-demand from archived transactions

### Archiving Strategy
- **Scheduled Jobs**:
  - Weekly archive jobs using AWS Lambda
  - Incremental archiving to minimize processing
  - Stored in S3 Standard-IA storage class
- **Data Format**:
  - Parquet files for efficient columnar storage
  - JSON bulk export as alternative
  - Structured by time periods (monthly archives)

### Retrieval Mechanism
- **Real-time Data**: Direct database queries
- **Archived Data**:
  - On-demand retrieval from S3
  - Lambda-based processing
  - Results caching in DynamoDB

## Query Optimization

### Free Tier Performance Strategies
- **Efficient Queries**:
  - Index-aware query design
  - Projection expressions (retrieve only needed fields)
  - Avoiding table scans in DynamoDB
- **Caching Layer**:
  - In-memory caching within Lambda
  - DynamoDB result caching with TTL
  - Invalidation strategies

### Data Aggregation
- **Pre-computed Aggregates**:
  - Daily/weekly/monthly summaries
  - Category-based aggregations
  - Materialized views concept
- **Lazy Loading**:
  - Fetch detailed data only when needed
  - Progressive enhancement approach
  - Client-side aggregation where possible

## Backup and Recovery

### Free Tier Backup Strategy
- **Manual Exports**:
  - Scheduled full database exports
  - Stored in S3 with lifecycle policies
  - Rotational backup schedule
- **Point-in-Time Recovery**:
  - Transaction logs in S3
  - Incremental backup approach
  - Recovery runbooks

### Disaster Recovery
- **Recovery Process**:
  - Database recreation from S3 backups
  - Lambda-based restoration
  - Verification procedures
- **Testing**:
  - Monthly recovery testing
  - Data integrity validation
  - Performance verification

## Data Migration Strategy

### Initial Data Loading
- **CSV/JSON Import**:
  - Batched imports to stay within free tier limits
  - Progressive loading with validation
  - Error handling and recovery
- **Vector Generation**:
  - Phased embedding creation
  - Prioritization of recent transactions
  - Background processing

### Schema Evolution
- **Versioning Strategy**:
  - Schema version tracking
  - Backward compatibility
  - Migration scripts
- **Migration Approach**:
  - In-place updates where possible
  - Copy-and-switch for major changes
  - Blue/green deployment for critical migrations

## Monitoring and Metrics

### Free Tier Monitoring
- **CloudWatch Metrics** (free tier):
  - Database operation counts
  - Storage utilization
  - Error rates
  - Performance metrics
- **Custom Logging**:
  - Operation timing logs
  - Error details
  - Usage patterns

### Alerts and Thresholds
- **Free Tier Limits**:
  - Storage approaching capacity
  - Rate limits nearing breach
  - Performance degradation
- **Notification Methods**:
  - Email alerts
  - CloudWatch alarms (free tier)
  - GitHub issue creation

## Optimization for Free Tier Longevity

### Storage Efficiency
- **Data Compression**:
  - Text field compression
  - Numeric optimization
  - Enum usage over strings
- **Binary Storage**:
  - BSON for MongoDB
  - Binary attributes for DynamoDB
  - Compact serialization formats

### Query Efficiency
- **Batch Operations**:
  - Bulk reads and writes
  - Transaction batching
  - Optimized connection usage
- **Connection Pooling**:
  - Reuse connections
  - Serverless connection management
  - Connection timeout optimization

## MongoDB Atlas Free Tier Management

The Finance RAG application uses MongoDB Atlas M0 free tier, which provides:
- 512MB storage limit
- Shared RAM and vCPU
- 100 operations per second
- 100 connections

### Database Schema Optimization

We've optimized our MongoDB collections to stay within the 512MB limit:

1. **Transactions Collection**:
   - Indexed fields: `date`, `category`, `description` (text index)
   - Data compression enabled
   - Selective fields stored (avoiding redundancy)
   - Estimated size: ~350KB per 1,000 transactions

2. **Categories Collection**:
   - Lightweight schema
   - Minimal metadata
   - Estimated size: <5KB

3. **Vector Embeddings Collection**:
   - Dimension-reduced embeddings (from 1536 to 384)
   - Sparse vector representation
   - Quantized values
   - Estimated size: ~150KB per 1,000 transactions

4. **Query Cache Collection**:
   - TTL index (24-hour expiration)
   - Compressed response storage
   - LRU-based cache management
   - Estimated size: ~50KB for typical usage

5. **User Feedback Collection**:
   - Minimal schema
   - Estimated size: <10KB for hundreds of feedback entries

### Data Retention Policy

To stay within the 512MB limit, we implement the following retention strategy:

1. **Transaction Aging**:
   - Transactions older than 24 months are archived
   - Summary statistics maintained for historical data
   - User can request restore of archived data

2. **Cache Management**:
   - Query cache automatic 24-hour expiration
   - LRU eviction for cached responses
   - Automatic purging when storage reaches 80% threshold

3. **Vector Store Pruning**:
   - Keep embeddings only for most recent 12 months by default
   - Dynamic reduction based on usage patterns
   - On-demand generation of embeddings for older transactions

4. **Feedback Data**:
   - Aggregate feedback older than 3 months
   - Store only statistical summaries for older feedback

### Data Rotation Implementation

```python
# Example of the data rotation script (simplified)
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient

def rotate_transaction_data(mongodb_uri):
    client = MongoClient(mongodb_uri)
    db = client.get_default_database()
    
    # Archive old transactions
    cutoff_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    old_transactions = list(db.transactions.find({"date": {"$lt": cutoff_date}}))
    
    # Create summary statistics
    if old_transactions:
        summary = {
            "period_start": min(t["date"] for t in old_transactions),
            "period_end": max(t["date"] for t in old_transactions),
            "count": len(old_transactions),
            "total_amount": sum(t["amount"] for t in old_transactions),
            "categories": {},
            "archived_date": datetime.now()
        }
        
        # Summarize by category
        for txn in old_transactions:
            category = txn.get("category", "Uncategorized")
            if category not in summary["categories"]:
                summary["categories"][category] = {
                    "count": 0,
                    "total": 0
                }
            summary["categories"][category]["count"] += 1
            summary["categories"][category]["total"] += txn["amount"]
        
        # Store summary and delete old transactions
        db.transaction_summaries.insert_one(summary)
        db.transactions.delete_many({"date": {"$lt": cutoff_date}})
        
    # Prune query cache beyond TTL
    db.query_cache.delete_many({
        "created_at": {"$lt": datetime.now() - timedelta(hours=24)}
    })
    
    # Clean up vector store for deleted transactions
    transaction_ids = [t["_id"] for t in db.transactions.find({}, {"_id": 1})]
    db.vector_embeddings.delete_many({"transaction_id": {"$nin": transaction_ids}})
    
    # Get current database stats
    db_stats = db.command("dbStats")
    current_size_mb = db_stats["dataSize"] / (1024 * 1024)
    
    return {
        "rotated_transactions": len(old_transactions),
        "current_storage_mb": current_size_mb,
        "storage_limit_mb": 512,
        "available_mb": 512 - current_size_mb
    }
```

## Vector Store Management

### Optimized Vector Storage

The Finance RAG application uses an optimized approach for vector embeddings:

1. **Dimension Reduction**:
   - Using Principal Component Analysis (PCA) to reduce embedding dimensions
   - Original: 1536 dimensions (OpenAI embeddings)
   - Reduced: 384 dimensions with minimal information loss
   - Storage reduction: ~75%

2. **Quantization**:
   - 32-bit to 8-bit quantization for vector values
   - Balanced approach between precision and storage
   - Storage reduction: ~75%

3. **Sparse Representation**:
   - Storing only non-zero elements for sparse vectors
   - Typically reduces storage by 30-50% for text embeddings

4. **Batched Processing**:
   - Vectors are processed in small batches to minimize memory usage
   - Background processing during low-activity periods

## AWS Free Tier Data Storage

### S3 Storage Strategy (Free Tier: 5GB)

The application uses AWS S3 for:
- Archived transaction data (compressed JSON)
- Backup exports (monthly)
- Static frontend assets

Storage is managed to stay under the 5GB free tier limit:
- Automated lifecycle policies
- Compression of all stored data
- Removal of unused backups

### CloudWatch Logs (Free Tier: 5GB)

Log management to stay within free tier:
- Structured logging with minimal verbosity
- Log expiration policies (7-day retention)
- Sampling for high-volume logs
- Aggregation of similar log events

## Backup and Recovery

### MongoDB Atlas Backup

MongoDB Atlas M0 free tier doesn't include automated backups, so we implement:

1. **Custom Backup Solution**:
   - Weekly full database exports to S3
   - Daily incremental backups using change streams
   - Retention of 4 weekly backups (rolling)

2. **Export Script Automation**:
   - Serverless function triggered on schedule
   - Compressed BSON/JSON exports
   - Backup verification and integrity checks

3. **Restoration Process**:
   - Documented restoration procedure
   - Test restores performed monthly
   - Ability to selectively restore collections

## Monitoring and Alerts

### Storage Monitoring

1. **Database Size Alerts**:
   - Threshold alerts at 70%, 80%, and 90% of free tier limit
   - Daily usage reports
   - Trend analysis for growth prediction

2. **Automated Responses**:
   - Automatic data rotation when thresholds are reached
   - Notification to administrators
   - Optional aggressive cleanup for emergency situations 