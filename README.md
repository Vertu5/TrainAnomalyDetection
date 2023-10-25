# Railway Anomaly Detection and Analysis

This project focuses on the detection and analysis of anomalies in diesel train data, with the aim of ensuring the safe and reliable operation of railway systems. The project utilizes data mining and machine learning techniques to identify and categorize anomalies, providing insights that can help prevent train incidents.

## Table of Contents
- [Overview](#overview)
- [Data Description](#data-description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Anomaly Detection](#anomaly-detection)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## Overview
The National Railway Company of Belgium (SNCB) is responsible for operating the rail service in Belgium. This project is sponsored by the rolling stock team of SNCB and aims to analyze real-life time-series data from diesel trains to detect and categorize anomalies. The data includes information on temperatures, pressures, RPMs, and GPS locations.

## Data Description
The dataset contains real-life errors and inconsistencies, including GPS positions reporting zeros and temperature values at exactly zero. The data spans from January 2023 to September 2023, allowing the exploration of the impact of weather conditions on train operations.

## Features
- Anomaly detection using various data mining methods.
- Enrichment of data with external weather data.
- Development of a comprehensive anomaly dashboard.
- Support for deployment in streaming mode.

## Project Structure
The project is organized into the following sections:
- **Data Loading and Preprocessing**: The raw data is loaded and preprocessed to make it suitable for analysis.
- **Enrichment with Weather Data**: External weather data is integrated to better understand the impact of weather conditions on anomalies.
- **Anomaly Detection Methods**: Multiple anomaly detection methods are implemented and compared.
- **Anomaly Dashboard**: A dashboard is developed to visualize anomalies and their relationship with weather, time, and location.

## Getting Started
To get started with the project, follow these steps:
1. Clone the repository to your local machine.
2. Install the required libraries and dependencies.
3. Preprocess the data using the provided code.
4. Run the anomaly detection methods.
5. Explore the anomaly dashboard for insights.

## Data Preprocessing
Data preprocessing includes data type conversion, handling missing values, and cleaning the dataset to prepare it for analysis. This step ensures the data is in a usable format.

## Anomaly Detection
The project employs various anomaly detection methods to identify and categorize anomalies. These methods are compared to determine their effectiveness in capturing anomalies in the train data.

## Dashboard
The dashboard provides a visual representation of anomalies, helping the SNCB rolling stock team understand patterns related to weather, time, and location. It enhances anomaly interpretation and decision-making.

## Contributing
We welcome contributions to this project. If you'd like to contribute, please follow our [contribution guidelines](CONTRIBUTING.md).

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for your own projects.

---

**Disclaimer:** This project is for educational and research purposes. It is not intended for production use in railway systems.

