require('./db/mongoose');
const axios = require('axios');
const Movie = require('./models/movie');

//const EPOCH = new Date(2023, 12, 31).getTime();

const API_KEY = '';

const genre = {
  28: 'Action',
  12: 'Adventure',
  16: 'Animation',
  35: 'Comedy',
  80: 'Crime',
  99: 'Documentary',
  18: 'Drama',
  10751: 'Family',
  14: 'Fantasy',
  36: 'History',
  27: 'Horror',
  10402: 'Music',
  9648: 'Mystery',
  10749: 'Romance',
  878: 'SciFi',
  53: 'Thriller',
  10752: 'War',
};

const requests = {
  trending: `https://api.themoviedb.org/3/trending/movie/week?api_key=${API_KEY}`,
  upcoming: `https://api.themoviedb.org/3/movie/upcoming?api_key=${API_KEY}&language=en-US`,
  popular: `https://api.themoviedb.org/3/movie/popular?api_key=${API_KEY}&language=en-US`,
  topRated: `https://api.themoviedb.org/3/movie/top_rated?api_key=${API_KEY}&language=en-US`,
  genre: `https://api.themoviedb.org/3/discover/movie?api_key=${API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&with_watch_monetization_types=flatrate`,
};

const preprocess = (movie, credits) => {
  const movieDetail = {
    id: movie['id'],
    original_title: movie['title'],
    overview: movie['overview'].toLowerCase(),
  };

  // title 
  movieDetail['title'] = movie['title'].toLowerCase().split(/[:]/)[0].trim();

  // adult
  movieDetail['adult'] = movie['adult'] ? 1 : 0;

  // vote_average
  movieDetail['vote_average'] = movie['vote_average'] || 5;

  // genre
  const genreIds = movie['genre_ids'];
  let genres = [];
  for (let id of genreIds) {
    if (genre[id]) genres.push(genre[id]);
  }
  movieDetail['genre'] = genres;

  // // release_date
  // const dateArray = movie['release_date'].split('-');
  // const date = new Date(+dateArray[0], +dateArray[1], +dateArray[2]);
  // const release_date = date.getTime() / EPOCH;
  // movieDetail['release_date'] = release_date;

  // cast
  let movieCast = '';
  const crew = credits.crew.filter((obj) => obj['job'] === 'Director');
  for (let dobj of crew) {
    if (dobj['name']) {
      movieCast += dobj['name'].replace(/\s+/g, '') + ' ';
    }
  }
  const cast = credits.cast;
  for (let i = 0; i < 3; i++) {
    if (cast[i]['name']) {
      movieCast += cast[i]['name'].replace(/\s+/g, '') + ' ';
    }
  }
  movieDetail['cast'] = movieCast.trim();

  return movieDetail;
};

const fetchMovieData = async (url) => {
  for (let i = 1; i <= 5; i++) {
    try {
      const response = await axios.get(`${url}&page=${i}`);
      const movies = response.data.results || response.data;

      for (let movie of movies) {
        if (movie['original_language'] !== 'en') {
          continue;
        }
        try {
          const response = await axios.get(
            `https://api.themoviedb.org/3/movie/${movie.id}/credits?api_key=${API_KEY}&language=en-US`
          );
          let credit = response.data.results || response.data;

          const movieDetail = preprocess(movie, credit);
          await new Movie(movieDetail).save();
          console.log('movie');
        } catch (err) {
          console.log('inside error\n' + err.message);
        }
      }
    } catch (err) {
      console.log('outside error\n' + err.message);
    }
  }
};

const setUpDatabase = async () => {
  // Trending
  await fetchMovieData(requests['trending']);

  // Upcoming 
  await fetchMovieData(requests['upcoming']);

  // Popular
  await fetchMovieData(requests['popular']);

  // Top Rated
  await fetchMovieData(requests['topRated']);

  // All Genre
  for (const gid in genre) {
    await fetchMovieData(`${requests['genre']}&with_genres=${genre[gid]}`);
  }

  console.log('final done');
};

setUpDatabase();
