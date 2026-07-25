from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Date, Numeric, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class DimDate(Base):
    """Date dimension table."""
    __tablename__ = 'dim_date'
    
    date_key = Column(Integer, primary_key=True)  # YYYYMMDD format
    full_date = Column(Date, nullable=False)
    day_of_week = Column(String(10))
    day_of_month = Column(Integer)
    month = Column(Integer)
    month_name = Column(String(10))
    quarter = Column(Integer)
    year = Column(Integer)
    is_weekend = Column(Boolean)
    is_holiday = Column(Boolean)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer)
    week_of_year = Column(Integer)

class DimCustomer(Base):
    """Customer dimension with SCD Type 2 support."""
    __tablename__ = 'dim_customer'
    
    customer_sk = Column(Integer, primary_key=True, autoincrement=True)  # Surrogate key
    customer_id = Column(String(20), nullable=False)  # Natural key
    customer_name = Column(String(100))
    email = Column(String(100))
    phone = Column(String(20))
    address = Column(String(200))
    city = Column(String(50))
    state = Column(String(50))
    zip_code = Column(String(10))
    country = Column(String(50))
    date_of_birth = Column(Date)
    gender = Column(String(10))
    income_bracket = Column(String(30))
    customer_segment = Column(String(30))  # Premium, Regular, Basic
    account_open_date = Column(Date)
    
    # SCD Type 2 columns
    effective_date = Column(Date, nullable=False)
    expiry_date = Column(Date)
    is_current = Column(Boolean, default=True)
    version = Column(Integer, default=1)
    
    __table_args__ = (
        Index('idx_customer_natural', 'customer_id', 'is_current'),
        Index('idx_customer_scd', 'customer_id', 'effective_date'),
    )

class DimProduct(Base):
    """Product dimension table."""
    __tablename__ = 'dim_product'
    
    product_sk = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(20), nullable=False)
    product_name = Column(String(100))
    product_category = Column(String(50))  # Savings, Checking, Credit Card, Loan, Investment
    product_type = Column(String(50))
    interest_rate = Column(Numeric(5, 2))
    annual_fee = Column(Numeric(10, 2))
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_product_natural', 'product_id'),
    )

class FactTransaction(Base):
    """Transaction fact table."""
    __tablename__ = 'fact_transactions'
    
    transaction_sk = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(30), nullable=False)
    
    # Foreign keys
    date_key = Column(Integer, ForeignKey('dim_date.date_key'), nullable=False)
    customer_sk = Column(Integer, ForeignKey('dim_customer.customer_sk'), nullable=False)
    product_sk = Column(Integer, ForeignKey('dim_product.product_sk'), nullable=False)
    
    # Measures
    amount = Column(Numeric(15, 2), nullable=False)
    balance_after = Column(Numeric(15, 2))
    transaction_type = Column(String(20))  # Credit, Debit, Transfer
    transaction_category = Column(String(50))  # Purchase, Withdrawal, Deposit, Fee, Interest
    channel = Column(String(20))  # Online, Mobile, ATM, Branch, POS
    merchant = Column(String(100))
    merchant_category = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime)
    is_fraudulent = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_fact_date', 'date_key'),
        Index('idx_fact_customer', 'customer_sk'),
        Index('idx_fact_product', 'product_sk'),
        Index('idx_fact_type', 'transaction_type'),
    )

class FactCustomerSnapshot(Base):
    """Monthly customer snapshot fact table."""
    __tablename__ = 'fact_customer_snapshot'
    
    snapshot_sk = Column(Integer, primary_key=True, autoincrement=True)
    date_key = Column(Integer, ForeignKey('dim_date.date_key'), nullable=False)
    customer_sk = Column(Integer, ForeignKey('dim_customer.customer_sk'), nullable=False)
    
    # Metrics
    total_transactions = Column(Integer)
    total_amount = Column(Numeric(15, 2))
    avg_transaction_amount = Column(Numeric(15, 2))
    account_balance = Column(Numeric(15, 2))
    num_products = Column(Integer)
    days_since_last_transaction = Column(Integer)
    is_active = Column(Boolean)
    
    __table_args__ = (
        Index('idx_snapshot_date', 'date_key'),
        Index('idx_snapshot_customer', 'customer_sk'),
    )
