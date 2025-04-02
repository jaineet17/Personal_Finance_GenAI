from src.embedding.vector_store import VectorStore; vs = VectorStore(); coll = vs.client.get_collection(vs.transaction_collection_name); count = coll.count(); print(f"Total transactions in vector store: {count}"); sample = coll.peek(5) if count > 0 else None; print("
Sample transactions:"); [print(f"{i+1}. {doc}") for i, doc in enumerate(sample.get("documents", []))] if sample else None;
