#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

house_data = {
 "price": 21.6,
 "crime_rate": 0.02731,
 "resid_area": 37.07,
 "air_qual": 0.469,
 "room_num": 6.421,
 "age": 78.9,
 "dist1": 4.99,
 "dist2": 4.7,
 "dist3": 5.12,
 "dist4": 5.06,
 "teachers": 22.2,
 "poor_prop": 9.14,
 "airport": "NO",
 "n_hos_beds": 7.332,
 "n_hot_rooms": 12.1728,
 "waterbody": "Lake",
 "rainfall": 42,
 "bus_ter": "YES",
 "parks": 0.046145633

}

house_data

response = requests.post(url, json = house_data).json()

print(response)

