const mongoose = require('mongoose');

mongoose.connect('mongodb://127.0.0.1:27017/ml_data', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});