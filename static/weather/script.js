const inputBox = document.querySelector('.input-box');
const searchBtn = document.getElementById('searchBtn');
const weather_img = document.querySelector('.weather-img');
const temperature = document.querySelector('.temperature');
const description = document.querySelector('.description');
const humidity = document.getElementById('humidity');
const wind_speed = document.getElementById('wind-speed');
const location_name = document.getElementById('location');
const feels_like = document.getElementById('feels-like');
const pressure = document.getElementById('pressure');
const visibility = document.getElementById('visibility');
const sunrise = document.getElementById('sunrise');
const sunset = document.getElementById('sunset');
const forecast_grid = document.getElementById('forecast-grid');

const location_not_found = document.querySelector('.location-not-found');
const weather_body = document.querySelector('.weather-body');

const api_key = "4c4286de4f6a3794841e570fd8bc4a0b";

async function checkWeather(city){
    const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${api_key}`;

    const weather_data = await fetch(`${url}`).then(response => response.json());

    if(weather_data.cod === `404`){
        location_not_found.style.display = "flex";
        weather_body.style.display = "none";
        console.log("error");
        return;
    }

    console.log("run");
    location_not_found.style.display = "none";
    weather_body.style.display = "block";
    
    // Update location and basic weather
    location_name.innerHTML = `${weather_data.name}, ${weather_data.sys.country}`;
    temperature.innerHTML = `${Math.round(weather_data.main.temp - 273.15)}<sup>°C</sup>`;
    description.innerHTML = `${weather_data.weather[0].description}`;

    // Update detailed weather info
    humidity.innerHTML = `${weather_data.main.humidity}%`;
    wind_speed.innerHTML = `${weather_data.wind.speed} m/s`;
    feels_like.innerHTML = `${Math.round(weather_data.main.feels_like - 273.15)}°C`;
    pressure.innerHTML = `${weather_data.main.pressure} hPa`;
    visibility.innerHTML = `${(weather_data.visibility / 1000).toFixed(1)} km`;
    sunrise.innerHTML = formatTime(weather_data.sys.sunrise);
    sunset.innerHTML = formatTime(weather_data.sys.sunset);

    const staticUrl = document.querySelector('script[src*="weather/script.js"]').src.replace('script.js', '');

    switch(weather_data.weather[0].main){
        case 'Clouds':
            weather_img.src = staticUrl + "img/cloud.png";
            break;
        case 'Clear':
            weather_img.src = staticUrl + "img/clear-sky.png";
            break;
        case 'Rain':
            weather_img.src = staticUrl + "img/rain.png";
            break;
        case 'Haze':
            weather_img.src = staticUrl + "img/haze.png";
            break;
        case 'Lightning':
            weather_img.src = staticUrl + "img/lightning.png";
            break;
        case 'Snow':
            weather_img.src = staticUrl + "img/snow.png";
            break;
        case 'Storm':
            weather_img.src = staticUrl + "img/storm.png";
            break;
        case 'Thunderstorm':
            weather_img.src = staticUrl + "img/thunderstorm.png";
            break;
        case 'Mist':
            weather_img.src = staticUrl + "img/mist.png";
            break;
    }

    // Fetch 5-day forecast
    await getForecast(weather_data.coord.lat, weather_data.coord.lon);

    console.log(weather_data);
}

async function getForecast(lat, lon){
    const forecast_url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${api_key}`;
    
    const forecast_data = await fetch(forecast_url).then(response => response.json());
    
    // Get one forecast per day (at 12:00)
    const dailyForecasts = forecast_data.list.filter(item => item.dt_txt.includes('12:00:00')).slice(0, 5);
    
    forecast_grid.innerHTML = '';
    
    dailyForecasts.forEach(day => {
        const date = new Date(day.dt * 1000);
        const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
        const temp = Math.round(day.main.temp - 273.15);
        const desc = day.weather[0].description;
        const icon = getWeatherIcon(day.weather[0].main);
        
        const forecastCard = `
            <div class="forecast-card">
                <div class="forecast-day">${dayName}</div>
                <img src="${icon}" alt="${desc}" class="forecast-icon">
                <div class="forecast-temp">${temp}°C</div>
                <div class="forecast-desc">${desc}</div>
            </div>
        `;
        
        forecast_grid.innerHTML += forecastCard;
    });
}

function getWeatherIcon(condition){
    const staticUrl = document.querySelector('script[src*="weather/script.js"]').src.replace('script.js', '');
    
    switch(condition){
        case 'Clouds':
            return staticUrl + "img/cloud.png";
        case 'Clear':
            return staticUrl + "img/clear-sky.png";
        case 'Rain':
            return staticUrl + "img/rain.png";
        case 'Haze':
            return staticUrl + "img/haze.png";
        case 'Snow':
            return staticUrl + "img/snow.png";
        case 'Thunderstorm':
            return staticUrl + "img/thunderstorm.png";
        case 'Mist':
            return staticUrl + "img/mist.png";
        default:
            return staticUrl + "img/cloud.png";
    }
}

function formatTime(unix){
    const date = new Date(unix * 1000);
    return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

searchBtn.addEventListener('click', ()=>{
    checkWeather(inputBox.value);
});

inputBox.addEventListener('keypress', (e)=>{
    if(e.key === 'Enter'){
        checkWeather(inputBox.value);
    }
});
