import sys
import os
import bson
import pymongo
from bson.son import SON
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import uuid
import json
from datetime import datetime
from pprint import pprint
import time
import numpy as np
from contextlib import contextmanager

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("BSON module location:", bson.__file__)
print("PyMongo version:", pymongo.__version__)
print("SON test:", SON([("a", 1), ("b", 2)]))

# Configuration from environment variables
def get_env_var(var_name, default=None, required=True):
    """Get environment variable with optional default and required check"""
    value = os.getenv(var_name, default)
    if required and value is None:
        print(f"Error: Required environment variable {var_name} is not set")
        sys.exit(1)
    return value

def get_env_bool(var_name, default=False):
    """Get boolean environment variable"""
    value = os.getenv(var_name, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_int(var_name, default):
    """Get integer environment variable"""
    try:
        return int(os.getenv(var_name, str(default)))
    except ValueError:
        print(f"Warning: Invalid integer value for {var_name}, using default: {default}")
        return default

# MongoDB Configuration
MONGO_URI = get_env_var("MONGO_URI")
MONGO_DB = get_env_var("MONGO_DB", "aufgang-pre-db")
MONGO_COLLECTION = get_env_var("MONGO_COLLECTION", "email_chunks")

# PostgreSQL Configuration
PG_HOST = get_env_var("PG_HOST")
PG_PORT = get_env_var("PG_PORT", "5443")
PG_DATABASE = get_env_var("PG_DATABASE")
PG_USER = get_env_var("PG_USER")
PG_PASSWORD = get_env_var("PG_PASSWORD")

# Migration settings - configurable via environment
BATCH_SIZE = get_env_int("BATCH_SIZE", 1000)
RETRY_MODE = get_env_bool("RETRY_MODE", False)
SKIP_DUPLICATES = get_env_bool("SKIP_DUPLICATES", True)
MAX_RETRIES = get_env_int("MAX_RETRIES", 3)
RETRY_DELAY = get_env_int("RETRY_DELAY", 5)
CONNECTION_TIMEOUT = get_env_int("CONNECTION_TIMEOUT", 30)

print(f"Configuration loaded:")
print(f"  MongoDB Database: {MONGO_DB}")
print(f"  MongoDB Collection: {MONGO_COLLECTION}")
print(f"  PostgreSQL Host: {PG_HOST}")
print(f"  PostgreSQL Port: {PG_PORT}")
print(f"  PostgreSQL Database: {PG_DATABASE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Retry Mode: {RETRY_MODE}")
print(f"  Skip Duplicates: {SKIP_DUPLICATES}")

# Connection parameters with better timeout handling
PG_CONNECTION_PARAMS = {
    'host': PG_HOST,
    'port': PG_PORT,
    'database': PG_DATABASE,
    'user': PG_USER,
    'password': PG_PASSWORD,
    'connect_timeout': CONNECTION_TIMEOUT,
    'keepalives_idle': 600,
    'keepalives_interval': 30,
    'keepalives_count': 3,
    'application_name': 'email_migration'
}

class DatabaseManager:
    def __init__(self):
        self.mongo_client = None
        self.pg_conn = None
        self.db = None
        self.collection = None
        
    def connect_mongodb(self):
        """Connect to MongoDB with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                if self.mongo_client:
                    self.mongo_client.close()
                    
                self.mongo_client = pymongo.MongoClient(MONGO_URI)
                self.db = self.mongo_client[MONGO_DB]
                self.collection = self.db[MONGO_COLLECTION]
                self.mongo_client.admin.command('ping')
                print(f"Successfully connected to MongoDB (attempt {attempt + 1})")
                return True
            except Exception as e:
                print(f"MongoDB connection attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return False
                time.sleep(RETRY_DELAY)
        return False
    
    def connect_postgresql(self):
        """Connect to PostgreSQL with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                if self.pg_conn:
                    try:
                        self.pg_conn.close()
                    except:
                        pass
                        
                self.pg_conn = psycopg2.connect(**PG_CONNECTION_PARAMS)
                self.pg_conn.autocommit = False
                
                # Test connection
                with self.pg_conn.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                    cursor.fetchone()
                    
                print(f"Successfully connected to PostgreSQL (attempt {attempt + 1})")
                return True
            except Exception as e:
                print(f"PostgreSQL connection attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return False
                time.sleep(RETRY_DELAY)
        return False
    
    def is_postgres_connected(self):
        """Check if PostgreSQL connection is alive"""
        if not self.pg_conn:
            return False
        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
            return True
        except Exception:
            return False
    
    def ensure_postgres_connection(self):
        """Ensure PostgreSQL connection is alive, reconnect if needed"""
        if not self.is_postgres_connected():
            print("PostgreSQL connection lost, reconnecting...")
            return self.connect_postgresql()
        return True
    
    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute database operation with retry on connection failure"""
        for attempt in range(MAX_RETRIES):
            try:
                if not self.ensure_postgres_connection():
                    raise Exception("Could not establish PostgreSQL connection")
                
                return operation(*args, **kwargs)
                
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                print(f"Database operation failed (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY)
            except Exception as e:
                # For other exceptions, don't retry
                raise e
    
    def close_connections(self):
        """Safely close all connections"""
        if self.pg_conn:
            try:
                self.pg_conn.close()
            except:
                pass
        if self.mongo_client:
            try:
                self.mongo_client.close()
            except:
                pass

# Initialize database manager
db_manager = DatabaseManager()

# Connect to databases
if not db_manager.connect_mongodb():
    print("Failed to connect to MongoDB after retries")
    sys.exit(1)

if not db_manager.connect_postgresql():
    print("Failed to connect to PostgreSQL after retries") 
    sys.exit(1)

# Enable pgvector extension
def enable_pgvector():
    with db_manager.pg_conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    db_manager.pg_conn.commit()
    print("pgvector extension enabled")

db_manager.execute_with_retry(enable_pgvector)

# Determine vector dimensions
print("Determining vector dimensions...")
sample_doc = db_manager.collection.find_one({"embedding": {"$exists": True}})
if sample_doc and 'embedding' in sample_doc:
    vector_dim = len(sample_doc['embedding'])
    print(f"Vector dimension: {vector_dim}")
else:
    print("No embedding found in sample document, defaulting to 1536 dimensions")
    vector_dim = 1536

# Table creation
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS email_chunks (
    id TEXT PRIMARY KEY,
    text TEXT,
    embedding vector({vector_dim}),
    email_id TEXT,
    user_id TEXT,
    sender TEXT,
    sender_name TEXT,
    to_names TEXT[],
    to_addresses TEXT[],
    cc_names TEXT[],
    cc_addresses TEXT[],
    subject TEXT,
    source TEXT,
    email TEXT,
    email_link TEXT,
    is_read BOOLEAN,
    is_sent BOOLEAN,
    starred BOOLEAN,
    hasAttachments BOOLEAN,
    email_content TEXT,
    email_created_date_time TIMESTAMPTZ
);
"""

def create_table():
    with db_manager.pg_conn.cursor() as cursor:
        cursor.execute(CREATE_TABLE_SQL)
    db_manager.pg_conn.commit()
    print("Table 'email_chunks' created/verified successfully")

db_manager.execute_with_retry(create_table)

# Keep your existing transformation functions
def convert_object_id_to_text(obj_id_str):
    if not obj_id_str:
        return None
    return str(obj_id_str)

def convert_date_field_exact(date_obj):
    """Convert MongoDB date to Python datetime for TIMESTAMPTZ"""
    if not date_obj:
        return None
    
    if isinstance(date_obj, datetime):
        if date_obj.tzinfo is None:
            from datetime import timezone
            return date_obj.replace(tzinfo=timezone.utc)
        return date_obj
    
    if isinstance(date_obj, dict) and '$date' in date_obj:
        try:
            return datetime.fromisoformat(date_obj['$date'].replace('Z', '+00:00'))
        except Exception:
            return None
    
    if isinstance(date_obj, str):
        try:
            return datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
        except Exception:
            return None
    
    return None

def convert_embedding_to_vector(embedding):
    if not embedding or not isinstance(embedding, list):
        return None
    try:
        np_array = np.array(embedding, dtype=np.float32)
        vector_str = '[' + ','.join(map(str, np_array.tolist())) + ']'
        return vector_str
    except Exception as e:
        print(f"Error converting embedding: {e}")
        return None

def transform_email_document(mongo_doc):
    try:
        transformed_doc = {}
        if '_id' in mongo_doc:
            transformed_doc['id'] = convert_object_id_to_text(str(mongo_doc['_id']))
        
        if 'user' in mongo_doc:
            transformed_doc['user_id'] = convert_object_id_to_text(str(mongo_doc['user']))
        elif 'user_id' in mongo_doc:
            transformed_doc['user_id'] = convert_object_id_to_text(str(mongo_doc['user_id']))
        
        if 'email_id' in mongo_doc:
            transformed_doc['email_id'] = convert_object_id_to_text(str(mongo_doc['email_id']))
        
        if 'text' in mongo_doc:
            transformed_doc['text'] = mongo_doc['text']
        
        if 'embedding' in mongo_doc:
            transformed_doc['embedding'] = convert_embedding_to_vector(mongo_doc['embedding'])
        
        email_fields = ['sender', 'sender_name', 'subject', 'source', 'email', 'email_link', 'email_content']
        for field in email_fields:
            if field in mongo_doc:
                transformed_doc[field] = mongo_doc[field]
        
        array_fields = ['to_names', 'to_addresses', 'cc_names', 'cc_addresses']
        for field in array_fields:
            if field in mongo_doc and mongo_doc[field]:
                transformed_doc[field] = mongo_doc[field]
            else:
                transformed_doc[field] = []
        
        boolean_fields = ['is_read', 'is_sent', 'starred']
        for field in boolean_fields:
            if field in mongo_doc:
                transformed_doc[field] = bool(mongo_doc[field])
        
        if 'hasAttachments' in mongo_doc:
            transformed_doc['hasAttachments'] = bool(mongo_doc['hasAttachments'])
        
        if 'email_created_date_time' in mongo_doc:
            transformed_doc['email_created_date_time'] = convert_date_field_exact(mongo_doc['email_created_date_time'])

        return transformed_doc
    
    except Exception as e:
        print(f"Error transforming document {mongo_doc.get('_id', 'unknown')}: {e}")
        print(f"Document data keys: {list(mongo_doc.keys()) if mongo_doc else 'None'}")
        return None

# Migration setup
total_docs = db_manager.collection.count_documents({})
print(f"Total documents in MongoDB: {total_docs}")

if total_docs == 0:
    print("No documents found in collection!")
    sys.exit(0)

# Insert statement
insert_sql = """
INSERT INTO email_chunks (
    id, text, embedding, email_id, user_id, sender, sender_name,
    to_names, to_addresses, cc_names, cc_addresses, subject, source,
    email, email_link, is_read, is_sent, starred, hasAttachments,
    email_content, email_created_date_time
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
"""
if SKIP_DUPLICATES:
    insert_sql += " ON CONFLICT (id) DO NOTHING"

def execute_batch_insert(batch_values):
    """Execute batch insert with connection retry"""
    def _do_insert():
        with db_manager.pg_conn.cursor() as pg_cursor:
            execute_batch(pg_cursor, insert_sql, batch_values, page_size=len(batch_values))
        db_manager.pg_conn.commit()
    
    db_manager.execute_with_retry(_do_insert)

# Migration with improved error handling
print("Starting migration process...")
start_time = time.time()
processed = 0
successful = 0
errors = 0
batch_values = []
last_processed_id = None

try:
    # Get last processed document for resume capability
    def get_last_processed_id():
        try:
            with db_manager.pg_conn.cursor() as cursor:
                cursor.execute("SELECT id FROM email_chunks ORDER BY id DESC LIMIT 1")
                result = cursor.fetchone()
                return result[0] if result else None
        except:
            return None
    
    if RETRY_MODE:
        last_processed_id = db_manager.execute_with_retry(get_last_processed_id)
        if last_processed_id:
            print(f"Resuming from last processed ID: {last_processed_id}")
    
    # Create query for resuming
    query = {}
    if last_processed_id and RETRY_MODE:
        query = {"_id": {"$gt": bson.ObjectId(last_processed_id)}}
    
    migration_cursor = db_manager.collection.find(query)
    
    for mongo_doc in migration_cursor:
        try:
            transformed_doc = transform_email_document(mongo_doc)
            if transformed_doc is None:
                errors += 1
                processed += 1
                continue
            
            values = (
                transformed_doc.get('id'),
                transformed_doc.get('text'),
                transformed_doc.get('embedding'),
                transformed_doc.get('email_id'),
                transformed_doc.get('user_id'),
                transformed_doc.get('sender'),
                transformed_doc.get('sender_name'),
                transformed_doc.get('to_names', []),
                transformed_doc.get('to_addresses', []),
                transformed_doc.get('cc_names', []),
                transformed_doc.get('cc_addresses', []),
                transformed_doc.get('subject'),
                transformed_doc.get('source'),
                transformed_doc.get('email'),
                transformed_doc.get('email_link'),
                transformed_doc.get('is_read'),
                transformed_doc.get('is_sent'),
                transformed_doc.get('starred'),
                transformed_doc.get('hasAttachments'),
                transformed_doc.get('email_content'),
                transformed_doc.get('email_created_date_time')
            )
            
            batch_values.append(values)
            processed += 1
            
            if len(batch_values) >= BATCH_SIZE:
                try:
                    execute_batch_insert(batch_values)
                    successful += len(batch_values)
                    print(f"Processed batch: {processed}/{total_docs} documents ({(processed/total_docs)*100:.1f}%)")
                except Exception as e:
                    print(f"Batch insert failed: {e}")
                    errors += len(batch_values)
                batch_values = []
                
        except Exception as e:
            print(f"Error processing document {mongo_doc.get('_id', 'unknown')}: {e}")
            errors += 1
    
    # Process remaining batch
    if batch_values:
        try:
            execute_batch_insert(batch_values)
            successful += len(batch_values)
            print(f"Processed final batch: {processed}/{total_docs} documents (100.0%)")
        except Exception as e:
            print(f"Final batch insert failed: {e}")
            errors += len(batch_values)

except KeyboardInterrupt:
    print("\nMigration interrupted by user")
except Exception as e:
    print(f"Unexpected error during migration: {e}")
finally:
    migration_cursor.close()

# Migration summary
end_time = time.time()
duration = end_time - start_time

print("\n" + "="*60)
print(f"EMAIL CHUNKS MIGRATION SUMMARY")
print("="*60)
print(f"Documents processed: {processed}")
print(f"Successfully migrated: {successful}")
print(f"Errors: {errors}")
print(f"Duration: {duration:.2f} seconds")
if duration > 0:
    print(f"Average speed: {processed/duration:.1f} docs/second")

# Close connections
db_manager.close_connections()
print("\nMigration completed!")



