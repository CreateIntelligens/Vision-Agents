from typing import Dict, Any

import httpx


async def get_weather_by_location(location: str) -> Dict[str, Any]:
    """
    Get current weather for a location using Open-Meteo API.

    Args:
        location: Name of the location (city, place, etc.)

    Returns:
        Weather data dictionary containing current weather information

    Raises:
        ValueError: If location not found or API response is invalid
        httpx.HTTPError: If API request fails
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Enhance location name for better geocoding results (especially for Taiwan)
        search_location = location

        # If it's a Taiwan district/area name, try adding "台北" or searching Taiwan
        taiwan_districts = ["內湖", "信義", "松山", "大安", "中山", "萬華", "中正", "大同", "南港", "文山", "士林", "北投"]
        if location in taiwan_districts:
            search_location = f"台北{location}"

        # Get geocoding data for the location
        geo_response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": search_location, "count": 5, "language": "zh"}
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        # If not found with enhanced name, try original location
        if not geo_data.get("results") and search_location != location:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 5}
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()

        if not geo_data.get("results"):
            raise ValueError(f"Location '{location}' not found")

        # Get weather for the location
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        return weather_data
