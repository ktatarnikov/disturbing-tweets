#!/usr/bin/env bash

python3 -m data.process_data data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
