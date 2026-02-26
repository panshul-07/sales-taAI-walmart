CREATE TABLE IF NOT EXISTS sales (
  id SERIAL PRIMARY KEY,
  date DATE NOT NULL,
  store_id INTEGER NOT NULL,
  product_id VARCHAR(64),
  category VARCHAR(128),
  quantity DOUBLE PRECISION NOT NULL,
  revenue DOUBLE PRECISION NOT NULL,
  cost DOUBLE PRECISION NOT NULL,
  profit DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS stores (
  store_id INTEGER PRIMARY KEY,
  location VARCHAR(128),
  size VARCHAR(64),
  store_type VARCHAR(64),
  region VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id SERIAL PRIMARY KEY,
  session_id VARCHAR(128) NOT NULL,
  role VARCHAR(16) NOT NULL,
  content TEXT NOT NULL,
  meta_json TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_store_date ON sales(store_id, date);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_chat_session_time ON chat_messages(session_id, created_at);
