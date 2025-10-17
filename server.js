// server.js - Main Backend Server
const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const nodemailer = require('nodemailer');
const cron = require('node-cron');
const { spawn } = require('child_process');
require('dotenv').config();

const app = express();

// Middleware
app.use(express.json());
app.use(cors());
app.use(express.static('public'));

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/Data', {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => {
  console.log('✓ Connected to MongoDB');
}).catch(err => {
  console.error('MongoDB connection error:', err);
});

// ==================== SCHEMAS ====================

// User Schema (Admin & Customer)
const userSchema = new mongoose.Schema({
  fullName: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: { type: String, enum: ['admin', 'customer'], required: true },
  phone: String,
  address: String,
  registeredAt: { type: Date, default: Date.now },
  isActive: { type: Boolean, default: true }
});

// Product Schema
const productSchema = new mongoose.Schema({
  Product_ID: { type: String, required: true, unique: true },
  Product_Name: String,
  Category: String,
  Sub_Category: String,
  Brand: String,
  Supplier_ID: String,
  Supplier_Name: String,
  Lead_Time_Days: Number,
  Quantity_in_Stock: { type: Number, default: 0 },
  Reorder_Level: Number,
  Reorder_Quantity: Number,
  Stock_Status: String,
  Unit_Cost: Number,
  Unit_Price: Number,
  Discount_Available: Boolean,
  Discount_Percentage: { type: Number, default: 0 },
  GST_Percentage: Number,
  Final_Price: Number,
  Manufacture_Date: Date,
  Expiry_Date: Date,
  Days_to_Expiry: Number,
  Is_Expired: Boolean,
  Units_Sold_Last_Month: Number,
  Units_Returned: Number,
  Last_Sold_Date: Date,
  Shelf_Life_Days: Number,
  Age_of_Stock_Days: Number,
  Profit_Per_Unit: Number,
  Total_Value_in_Stock: Number,
  image_url: { type: String, default: '/images/default-product.jpg' },
  description: String,
  updatedAt: { type: Date, default: Date.now }
});

// Flash Sale Schema
const flashSaleSchema = new mongoose.Schema({
  title: { type: String, required: true },
  description: String,
  discount_percentage: { type: Number, required: true },
  start_date: { type: Date, required: true },
  end_date: { type: Date, required: true },
  products: [{ type: String }], // Product IDs
  is_active: { type: Boolean, default: true },
  created_by: String,
  created_at: { type: Date, default: Date.now }
});

// Order Schema
const orderSchema = new mongoose.Schema({
  order_id: { type: String, required: true, unique: true },
  customer_id: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  customer_email: String,
  items: [{
    product_id: String,
    product_name: String,
    quantity: Number,
    unit_price: Number,
    discount: Number,
    total: Number
  }],
  subtotal: Number,
  tax: Number,
  shipping: Number,
  total_amount: Number,
  shipping_address: {
    street: String,
    city: String,
    state: String,
    pincode: String,
    phone: String
  },
  payment_status: { type: String, default: 'pending' },
  order_status: { type: String, default: 'processing' },
  tracking_id: String,
  ordered_at: { type: Date, default: Date.now },
  delivered_at: Date,
  cancellation_requested: { type: Boolean, default: false },
  cancellation_reason: String
});

// Ticket Schema
const ticketSchema = new mongoose.Schema({
  ticket_id: { type: String, required: true, unique: true },
  customer_id: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  customer_email: String,
  subject: String,
  message: String,
  status: { type: String, default: 'open' },
  admin_response: String,
  created_at: { type: Date, default: Date.now },
  resolved_at: Date
});

// Feedback Schema
const feedbackSchema = new mongoose.Schema({
  customer_id: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  customer_email: String,
  product_id: String,
  rating: { type: Number, min: 1, max: 5 },
  comment: String,
  created_at: { type: Date, default: Date.now }
});

// Alert Notification Schema
const alertSchema = new mongoose.Schema({
  alert_type: { type: String, enum: ['expiry', 'low_stock', 'reorder'] },
  product_id: String,
  product_name: String,
  message: String,
  severity: { type: String, enum: ['critical', 'warning', 'info'] },
  is_read: { type: Boolean, default: false },
  created_at: { type: Date, default: Date.now }
});

// Models
const User = mongoose.model('User', userSchema);
const Product = mongoose.model('Product', productSchema);
const FlashSale = mongoose.model('FlashSale', flashSaleSchema);
const Order = mongoose.model('Order', orderSchema);
const Ticket = mongoose.model('Ticket', ticketSchema);
const Feedback = mongoose.model('Feedback', feedbackSchema);
const Alert = mongoose.model('Alert', alertSchema);

// ==================== MIDDLEWARE ====================

// Authentication Middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key', (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// Admin Only Middleware
const adminOnly = (req, res, next) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Admin access required' });
  }
  next();
};

// ==================== EMAIL CONFIGURATION ====================

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS
  }
});

const sendEmail = async (to, subject, html) => {
  try {
    await transporter.sendMail({
      from: process.env.EMAIL_USER,
      to,
      subject,
      html
    });
    return true;
  } catch (error) {
    console.error('Email error:', error);
    return false;
  }
};

// ==================== AUTH ROUTES ====================

// Register Customer
app.post('/api/auth/register/customer', async (req, res) => {
  try {
    const { fullName, email, password, phone, address } = req.body;

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const user = new User({
      fullName,
      email,
      password: hashedPassword,
      role: 'customer',
      phone,
      address
    });

    await user.save();

    // Send welcome email
    await sendEmail(email, 'Welcome to Cosmetics Store', `
      <h2>Welcome ${fullName}!</h2>
      <p>Thank you for registering with us.</p>
    `);

    res.status(201).json({ message: 'Registration successful', userId: user._id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Register Admin (with secret code)
app.post('/api/auth/register/admin', async (req, res) => {
  try {
    const { fullName, email, password, secretCode } = req.body;

    // Verify secret code
    if (secretCode !== process.env.ADMIN_SECRET_CODE || secretCode !== 'COSM2025ADMIN') {
      return res.status(403).json({ error: 'Invalid secret code' });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const user = new User({
      fullName,
      email,
      password: hashedPassword,
      role: 'admin'
    });

    await user.save();

    res.status(201).json({ message: 'Admin registration successful', userId: user._id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password, role } = req.body;

    const user = await User.findOne({ email, role, isActive: true });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign(
      { userId: user._id, email: user.email, role: user.role },
      process.env.JWT_SECRET || 'your-secret-key',
      { expiresIn: '24h' }
    );

    res.json({
      token,
      user: {
        id: user._id,
        fullName: user.fullName,
        email: user.email,
        role: user.role
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== PRODUCT ROUTES ====================

// Get all products (with filters)
app.get('/api/products', async (req, res) => {
  try {
    const { category, search, sort, inStock } = req.query;
    let query = {};

    if (category && category !== 'all') {
      query.Category = category;
    }

    if (search) {
      query.$or = [
        { Product_Name: { $regex: search, $options: 'i' } },
        { Brand: { $regex: search, $options: 'i' } },
        { Sub_Category: { $regex: search, $options: 'i' } }
      ];
    }

    if (inStock === 'true') {
      query.Quantity_in_Stock = { $gt: 0 };
    }

    let products = await Product.find(query);

    // Sorting
    if (sort === 'price_low') {
      products = products.sort((a, b) => a.Final_Price - b.Final_Price);
    } else if (sort === 'price_high') {
      products = products.sort((a, b) => b.Final_Price - a.Final_Price);
    } else if (sort === 'name') {
      products = products.sort((a, b) => a.Product_Name.localeCompare(b.Product_Name));
    }

    res.json(products);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get single product
app.get('/api/products/:id', async (req, res) => {
  try {
    const product = await Product.findOne({ Product_ID: req.params.id });
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }
    res.json(product);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Add product (Admin only)
app.post('/api/products', authenticateToken, adminOnly, async (req, res) => {
  try {
    const product = new Product(req.body);
    await product.save();
    res.status(201).json({ message: 'Product added successfully', product });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update product (Admin only)
app.put('/api/products/:id', authenticateToken, adminOnly, async (req, res) => {
  try {
    const product = await Product.findOneAndUpdate(
      { Product_ID: req.params.id },
      { ...req.body, updatedAt: Date.now() },
      { new: true }
    );
    
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }

    res.json({ message: 'Product updated successfully', product });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Delete product (Admin only)
app.delete('/api/products/:id', authenticateToken, adminOnly, async (req, res) => {
  try {
    const product = await Product.findOneAndDelete({ Product_ID: req.params.id });
    
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }

    res.json({ message: 'Product deleted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get categories
app.get('/api/categories', async (req, res) => {
  try {
    const categories = await Product.distinct('Category');
    res.json(categories);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== CONTINUE IN NEXT PART ====================

// ==================== FLASH SALE ROUTES ====================

// Create Flash Sale (Admin only)
app.post('/api/flash-sales', authenticateToken, adminOnly, async (req, res) => {
  try {
    const { title, description, discount_percentage, start_date, end_date, products } = req.body;

    const flashSale = new FlashSale({
      title,
      description,
      discount_percentage,
      start_date,
      end_date,
      products,
      created_by: req.user.email
    });

    await flashSale.save();

    // Update product prices
    // Update products with discount
    for (const productId of products) {
      const product = await Product.findOne({ Product_ID: productId });
      if (product) {
        product.Discount_Available = true;
        product.Discount_Percentage = discount_percentage;
        product.Final_Price = product.Unit_Price * (1 - discount_percentage/100) * (1 + product.GST_Percentage/100);
        await product.save();
      }
    }

    // Notify customers
    const customers = await User.find({ role: 'customer', isActive: true });
    for (const customer of customers) {
      await sendEmail(customer.email, `Flash Sale: ${title}`, `
        <h2>🎉 ${title}</h2>
        <p>${description}</p>
        <p><strong>${discount_percentage}% OFF</strong></p>
        <p>Valid from ${new Date(start_date).toLocaleDateString()} to ${new Date(end_date).toLocaleDateString()}</p>
        <a href="${process.env.FRONTEND_URL}/products">Shop Now</a>
      `);
    }

    res.status(201).json({ message: 'Flash sale created successfully', flashSale });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all flash sales
app.get('/api/flash-sales', async (req, res) => {
  try {
    const now = new Date();
    const flashSales = await FlashSale.find({
      is_active: true,
      start_date: { $lte: now },
      end_date: { $gte: now }
    });
    res.json(flashSales);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== ORDER ROUTES ====================

// Create Order
app.post('/api/orders', authenticateToken, async (req, res) => {
  try {
    const { items, shipping_address } = req.body;
    
    // Generate order ID
    const order_id = `ORD${Date.now()}${Math.floor(Math.random() * 1000)}`;
    
    // Calculate totals
    let subtotal = 0;
    const orderItems = [];

    for (const item of items) {
      const product = await Product.findOne({ Product_ID: item.product_id });
      
      if (!product) {
        return res.status(404).json({ error: `Product ${item.product_id} not found` });
      }

      if (product.Quantity_in_Stock < item.quantity) {
        return res.status(400).json({ error: `Insufficient stock for ${product.Product_Name}` });
      }

      const itemTotal = product.Final_Price * item.quantity;
      subtotal += itemTotal;

      orderItems.push({
        product_id: product.Product_ID,
        product_name: product.Product_Name,
        quantity: item.quantity,
        unit_price: product.Unit_Price,
        discount: product.Discount_Percentage,
        total: itemTotal
      });

      // Update stock
      product.Quantity_in_Stock -= item.quantity;
      product.Units_Sold_Last_Month += item.quantity;
      product.Last_Sold_Date = new Date();
      await product.save();
    }

    const shipping = subtotal > 500 ? 0 : 50;
    const tax = subtotal * 0.18; // 18% GST
    const total_amount = subtotal + shipping + tax;

    const order = new Order({
      order_id,
      customer_id: req.user.userId,
      customer_email: req.user.email,
      items: orderItems,
      subtotal,
      tax,
      shipping,
      total_amount,
      shipping_address,
      tracking_id: `TRK${Date.now()}`
    });

    await order.save();

    // Send order confirmation email
    await sendEmail(req.user.email, `Order Confirmation - ${order_id}`, `
      <h2>Order Confirmed!</h2>
      <p>Your order ${order_id} has been placed successfully.</p>
      <p><strong>Total Amount:</strong> ₹${total_amount.toFixed(2)}</p>
      <p><strong>Tracking ID:</strong> ${order.tracking_id}</p>
      <p>We'll notify you when your order is shipped.</p>
    `);

    res.status(201).json({ message: 'Order placed successfully', order });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get customer orders
app.get('/api/orders/my-orders', authenticateToken, async (req, res) => {
  try {
    const orders = await Order.find({ customer_id: req.user.userId }).sort({ ordered_at: -1 });
    res.json(orders);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all orders (Admin)
app.get('/api/orders', authenticateToken, adminOnly, async (req, res) => {
  try {
    const orders = await Order.find().sort({ ordered_at: -1 });
    res.json(orders);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update order status (Admin)
app.put('/api/orders/:orderId/status', authenticateToken, adminOnly, async (req, res) => {
  try {
    const { order_status } = req.body;
    const order = await Order.findOne({ order_id: req.params.orderId });

    if (!order) {
      return res.status(404).json({ error: 'Order not found' });
    }

    order.order_status = order_status;
    if (order_status === 'delivered') {
      order.delivered_at = new Date();
    }
    await order.save();

    // Notify customer
    await sendEmail(order.customer_email, `Order Update - ${order.order_id}`, `
      <h2>Order Status Updated</h2>
      <p>Your order ${order.order_id} is now: <strong>${order_status}</strong></p>
      <p>Tracking ID: ${order.tracking_id}</p>
    `);

    res.json({ message: 'Order status updated', order });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Cancel order request
app.post('/api/orders/:orderId/cancel', authenticateToken, async (req, res) => {
  try {
    const { cancellation_reason } = req.body;
    const order = await Order.findOne({ order_id: req.params.orderId, customer_id: req.user.userId });

    if (!order) {
      return res.status(404).json({ error: 'Order not found' });
    }

    if (order.order_status === 'delivered') {
      return res.status(400).json({ error: 'Cannot cancel delivered order' });
    }

    order.cancellation_requested = true;
    order.cancellation_reason = cancellation_reason;
    await order.save();

    res.json({ message: 'Cancellation request submitted' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== TICKET ROUTES ====================

// Create ticket
app.post('/api/tickets', authenticateToken, async (req, res) => {
  try {
    const { subject, message } = req.body;
    const ticket_id = `TKT${Date.now()}`;

    const ticket = new Ticket({
      ticket_id,
      customer_id: req.user.userId,
      customer_email: req.user.email,
      subject,
      message
    });

    await ticket.save();
    res.status(201).json({ message: 'Ticket created successfully', ticket });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get user tickets
app.get('/api/tickets/my-tickets', authenticateToken, async (req, res) => {
  try {
    const tickets = await Ticket.find({ customer_id: req.user.userId }).sort({ created_at: -1 });
    res.json(tickets);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all tickets (Admin)
app.get('/api/tickets', authenticateToken, adminOnly, async (req, res) => {
  try {
    const tickets = await Ticket.find().sort({ created_at: -1 });
    res.json(tickets);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Respond to ticket (Admin)
app.post('/api/tickets/:ticketId/respond', authenticateToken, adminOnly, async (req, res) => {
  try {
    const { response } = req.body;
    const ticket = await Ticket.findOne({ ticket_id: req.params.ticketId });

    if (!ticket) {
      return res.status(404).json({ error: 'Ticket not found' });
    }

    ticket.admin_response = response;
    ticket.status = 'resolved';
    ticket.resolved_at = new Date();
    await ticket.save();

    // Email customer
    await sendEmail(ticket.customer_email, `Ticket Response - ${ticket.ticket_id}`, `
      <h2>Ticket Update</h2>
      <p><strong>Subject:</strong> ${ticket.subject}</p>
      <p><strong>Response:</strong></p>
      <p>${response}</p>
    `);

    res.json({ message: 'Response sent successfully', ticket });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== FEEDBACK ROUTES ====================

// Submit feedback
app.post('/api/feedback', authenticateToken, async (req, res) => {
  try {
    const { product_id, rating, comment } = req.body;

    const feedback = new Feedback({
      customer_id: req.user.userId,
      customer_email: req.user.email,
      product_id,
      rating,
      comment
    });

    await feedback.save();
    res.status(201).json({ message: 'Feedback submitted successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get product feedback
app.get('/api/feedback/product/:productId', async (req, res) => {
  try {
    const feedback = await Feedback.find({ product_id: req.params.productId }).sort({ created_at: -1 });
    res.json(feedback);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get all feedback (Admin)
app.get('/api/feedback', authenticateToken, adminOnly, async (req, res) => {
  try {
    const feedback = await Feedback.find().sort({ created_at: -1 });
    res.json(feedback);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== ALERT ROUTES ====================

// Get alerts (Admin)
app.get('/api/alerts', authenticateToken, adminOnly, async (req, res) => {
  try {
    const alerts = await Alert.find().sort({ created_at: -1 }).limit(50);
    res.json(alerts);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Mark alert as read
app.put('/api/alerts/:id/read', authenticateToken, adminOnly, async (req, res) => {
  try {
    const alert = await Alert.findByIdAndUpdate(req.params.id, { is_read: true }, { new: true });
    res.json(alert);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== ANALYTICS ROUTES ====================

// Dashboard statistics (Admin)
app.get('/api/analytics/dashboard', authenticateToken, adminOnly, async (req, res) => {
  try {
    const totalProducts = await Product.countDocuments();
    const totalOrders = await Order.countDocuments();
    const totalRevenue = await Order.aggregate([
      { $group: { _id: null, total: { $sum: '$total_amount' } } }
    ]);

    const expiringProducts = await Product.countDocuments({ Days_to_Expiry: { $lte: 30, $gte: 0 } });
    const expiredProducts = await Product.countDocuments({ Is_Expired: true });
    const lowStockProducts = await Product.countDocuments({ $expr: { $lte: ['$Quantity_in_Stock', '$Reorder_Level'] } });

    const topSellingProducts = await Product.find().sort({ Units_Sold_Last_Month: -1 }).limit(10);

    const categoryWiseStock = await Product.aggregate([
      { $group: { _id: '$Category', total_stock: { $sum: '$Quantity_in_Stock' }, total_value: { $sum: '$Total_Value_in_Stock' } } }
    ]);

    res.json({
      totalProducts,
      totalOrders,
      totalRevenue: totalRevenue[0]?.total || 0,
      expiringProducts,
      expiredProducts,
      lowStockProducts,
      topSellingProducts,
      categoryWiseStock
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Expiry trend analysis
app.get('/api/analytics/expiry-trends', authenticateToken, adminOnly, async (req, res) => {
  try {
    const expiryTrends = await Product.aggregate([
      {
        $bucket: {
          groupBy: '$Days_to_Expiry',
          boundaries: [-Infinity, 0, 30, 90, 180, 365, Infinity],
          default: 'Other',
          output: {
            count: { $sum: 1 },
            products: { $push: '$Product_Name' }
          }
        }
      }
    ]);

    res.json(expiryTrends);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== ML PREDICTION ROUTE ====================

// Predict expiry for product
app.post('/api/predict-expiry', authenticateToken, adminOnly, async (req, res) => {
  try {
    const productData = req.body;

    // Call Python prediction script
    const python = spawn('python', ['prediction_script.py', JSON.stringify(productData)]);

    let result = '';
    python.stdout.on('data', (data) => {
      result += data.toString();
    });

    python.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });

    python.on('close', (code) => {
      if (code === 0) {
        try {
          const prediction = JSON.parse(result);
          res.json(prediction);
        } catch (e) {
          res.status(500).json({ error: 'Failed to parse prediction result' });
        }
      } else {
        res.status(500).json({ error: 'Prediction failed' });
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ==================== CRON JOBS ====================

// Daily check for expiring products and low stock
cron.schedule('0 0 * * *', async () => {
  console.log('Running daily inventory check...');

  try {
    // Check for near-expiry products
    const nearExpiryProducts = await Product.find({
      Days_to_Expiry: { $lte: 30, $gte: 0 },
      Quantity_in_Stock: { $gt: 0 }
    });

    for (const product of nearExpiryProducts) {
      // Create alert
      await Alert.create({
        alert_type: 'expiry',
        product_id: product.Product_ID,
        product_name: product.Product_Name,
        message: `${product.Product_Name} expires in ${product.Days_to_Expiry} days`,
        severity: product.Days_to_Expiry <= 7 ? 'critical' : 'warning'
      });

      // Auto-add to discount if not already
      if (!product.Discount_Available) {
        const discountPercent = product.Days_to_Expiry <= 7 ? 50 : 30;
        product.Discount_Available = true;
        product.Discount_Percentage = discountPercent;
        product.Final_Price = product.Unit_Price * (1 - discountPercent/100) * (1 + product.GST_Percentage/100);
        await product.save();
      }
    }

    // Check for low stock
    const lowStockProducts = await Product.find({
      $expr: { $lte: ['$Quantity_in_Stock', '$Reorder_Level'] }
    });

    for (const product of lowStockProducts) {
      await Alert.create({
        alert_type: 'low_stock',
        product_id: product.Product_ID,
        product_name: product.Product_Name,
        message: `${product.Product_Name} stock is below reorder level`,
        severity: 'warning'
      });
    }

    console.log('Daily check completed');
  } catch (error) {
    console.error('Cron job error:', error);
  }
});

// Update Days_to_Expiry daily
cron.schedule('0 1 * * *', async () => {
  console.log('Updating expiry days...');

  try {
    const products = await Product.find();
    for (const product of products) {
      if (product.Expiry_Date) {
        const today = new Date();
        const expiryDate = new Date(product.Expiry_Date);
        const daysToExpiry = Math.floor((expiryDate - today) / (1000 * 60 * 60 * 24));
        
        product.Days_to_Expiry = daysToExpiry;
        product.Is_Expired = daysToExpiry < 0;
        await product.save();
      }
    }
    console.log('Expiry days updated');
  } catch (error) {
    console.error('Update error:', error);
  }
});

// ==================== SERVER START ====================

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log(`✓ Server running on port ${PORT}`);
  console.log(`✓ MongoDB connected`);
  console.log(`✓ Cron jobs scheduled`);
  console.log('='.repeat(60));
});

module.exports = app;