-- sql/queries.sql -- Part A Task 6 SQL Analysis Queries
-- Database: data/instamart.db (SQLite 3.25+)

-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500 (WHERE), group by customer,
-- and keep customers whose total of those orders exceeds 8000 (HAVING). Reference: 30 customers.
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_large_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_large_spend DESC;

-- Q2: Three-table join
-- Revenue by city and product category (join orders, customers, products).
-- Reference: 30 rows; the 30 revenues add up to INR 2,998,668.80; largest cell is Bengaluru / Electronics at INR 233,909.45.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city ASC, p.category ASC;

-- Q3: Scalar subquery
-- Products priced above the average catalogue price. Reference: 11 products.
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id ASC;

-- Q4: LEFT JOIN
-- Customers who have no completed order. Reference: 25 customers.
SELECT
    c.customer_id,
    c.city,
    c.signup_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id ASC;

-- Q5: Common Table Expression (CTE)
-- One CTE computes each customer's total completed spend and a second computes the average of those totals;
-- the main query returns customers whose total is above the average.
-- Reference: 550 customers; average INR 2,552.06 (rounded to 2 decimals).
WITH customer_completed_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
benchmark_spend AS (
    SELECT
        AVG(total_spend) AS avg_spend
    FROM customer_completed_spend
)
SELECT
    ccs.customer_id,
    ROUND(ccs.total_spend, 2) AS total_spend,
    ROUND(bs.avg_spend, 2) AS benchmark_avg
FROM customer_completed_spend ccs
CROSS JOIN benchmark_spend bs
WHERE ccs.total_spend > bs.avg_spend
ORDER BY ccs.total_spend DESC;

-- Q6: Ranking window functions
-- For each customer with completed spend, compute total spend and then RANK(), DENSE_RANK() and ROW_NUMBER()
-- using OVER (PARTITION BY city ORDER BY total_spend DESC).
-- Reference: the top customer per city is Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40),
-- Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10);
-- exactly 8 rows have RANK() different from ROW_NUMBER().
WITH customer_city_spend AS (
    SELECT
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_city_spend
ORDER BY city ASC, total_spend DESC, customer_id ASC;

-- Q7: Running total
-- Daily completed revenue (the day is the first ten characters of order_ts) with a cumulative revenue column computed as SUM(revenue) OVER (ORDER BY day).
-- Reference: 183 daily rows; the last cumulative value is INR 2,998,668.80.
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    ROUND(SUM(revenue) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day ASC;

