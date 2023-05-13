const mongoose = require('mongoose');

const movieSchema = new mongoose.Schema(
  {
    id: {
      type: Number,
      unique: true,
    },
    adult: Number, 
    title: String,
    original_title: String,
    overview: String,
    genre: [String],
    vote_average: Number,
    cast: String,
  }
);

const Movie = mongoose.model('Movie', movieSchema);

module.exports = Movie;
