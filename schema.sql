-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(200) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stores table
CREATE TABLE stores (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Receipts table
CREATE TABLE receipts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    store_id INTEGER REFERENCES stores(id) ON DELETE SET NULL,
    receipt_key VARCHAR(100) NOT NULL,
    date_of_purchase DATE,
    total_amount NUMERIC(10, 2),
    receipt_image BYTEA,  -- store receipt image as binary data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (store_id, receipt_key)  -- enforce unique receipt per store by receipt_key
);

-- Receipt items table
CREATE TABLE receipt_items (
    id SERIAL PRIMARY KEY,
    receipt_id INTEGER REFERENCES receipts(id) ON DELETE CASCADE,
    item_name VARCHAR(200),
    quantity INTEGER,
    unit_price NUMERIC(10, 2),
    total_price NUMERIC(10, 2)
);

-- Labels table (for tagging receipts or items)
CREATE TABLE labels (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    receipt_id INTEGER REFERENCES receipts(id) ON DELETE CASCADE,
    label VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Corrections table
CREATE TABLE corrections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    receipt_item_id INTEGER REFERENCES receipt_items(id) ON DELETE CASCADE,
    
    original_name VARCHAR(200),
    corrected_name VARCHAR(200),
    
    original_quantity INTEGER,
    corrected_quantity INTEGER,
    
    original_unit_price NUMERIC(10, 2),
    corrected_unit_price NUMERIC(10, 2),
    
    original_total_price NUMERIC(10, 2),
    corrected_total_price NUMERIC(10, 2),
    
    correction_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
