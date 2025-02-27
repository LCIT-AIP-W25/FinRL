import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const LiveStockChart = () => {
  const [stockData, setStockData] = useState([]);
  const [labels, setLabels] = useState([]);

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get('https://api.stockdata.org/v1/data/intraday?symbols=AAPL&api_token=YOUR_API_KEY');
        const data = response.data.data;
        const prices = data.map(entry => entry.price);
        const times = data.map(entry => new Date(entry.timestamp).toLocaleTimeString());

        setStockData(prices);
        setLabels(times);
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
    const interval = setInterval(fetchStockData, 60000); // Fetch data every minute

    return () => clearInterval(interval);
  }, []);

  const chartData = {
    labels: labels,
    datasets: [
      {
        label: 'AAPL Stock Price',
        data: stockData,
        fill: false,
        backgroundColor: 'rgba(75,192,192,0.4)',
        borderColor: 'rgba(75,192,192,1)',
      },
    ],
  };

  return (
    <div>
      <h2>Live Stock Data</h2>
      <Line data={chartData} />
    </div>
  );
};

export default LiveStockChart;