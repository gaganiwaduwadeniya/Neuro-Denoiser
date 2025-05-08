const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/encoder-app', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);
    
    // Log info about the database max document size
    // MongoDB has a max document size of 16MB
    const maxDocumentSizeInMB = 16;
    console.log(`MongoDB max document size: ${maxDocumentSizeInMB}MB`);
    console.log(`Note: Images larger than ~10MB may need to be resized before storage to avoid exceeding limits`);
    
    return conn;
  } catch (error) {
    console.error(`Error connecting to MongoDB: ${error.message}`);
    process.exit(1);
  }
};

module.exports = connectDB; 