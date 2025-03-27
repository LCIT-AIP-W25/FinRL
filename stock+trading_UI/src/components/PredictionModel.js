import React, { useState } from 'react';
import styled from 'styled-components';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PredictionContainer = styled.div`
  padding: 20px;
  text-align: center;
`;

const PredictionButton = styled.button`
  background-color: #4CAF50;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;

  &:hover {
    background-color: #45a049;
  }
`;

const PredictionResult = styled.div`
  margin-top: 20px;
`;

const InputContainer = styled.div`
  margin-bottom: 20px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
`;

const Input = styled.input`
  padding: 10px;
  margin: 5px;
  border: 1px solid #ccc;
  border-radius: 5px;
  width: 300px; /* Adjust the width as needed */
`;

const Table = styled.table`
  margin: 20px auto;
  border-collapse: collapse;
  width: 80%;
`;

const Th = styled.th`
  border: 1px solid #ddd;
  padding: 8px;
  background-color: #f2f2f2;
`;

const Td = styled.td`
  border: 1px solid #ddd;
  padding: 8px;
`;

const PredictionModel = () => {
  const [prediction, setPrediction] = useState(null);
  const [stockName, setStockName] = useState('');

  const handlePrediction = async () => {
    try {
      const response = await fetch(`http://localhost:5000/predict?symbol=${stockName}`);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Failed to fetch:', error);
    }
  };

  const formatDataForChart = (data) => {
    const formattedData = data.last_10_days.map((item) => ({
      date: new Date(item.Date).toLocaleDateString(),
      close: item.Close,
    }));

    // Add the next day prediction separately
    const lastDate = new Date(data.last_10_days[data.last_10_days.length - 1].Date);
    const nextDayDate = new Date(lastDate);
    nextDayDate.setDate(lastDate.getDate() + 1);
    formattedData.push({
      date: nextDayDate.toLocaleDateString(),
      next_day_prediction: data.next_day_prediction,
    });

    return formattedData;
  };

  return (
    <PredictionContainer>
      <h2>Model Prediction</h2>
      <InputContainer>
        <Label htmlFor="stockName">Enter the Stock Name:</Label>
        <Input
          type="text"
          id="stockName"
          placeholder="Stock Name"
          value={stockName}
          onChange={(e) => setStockName(e.target.value)}
        />
      </InputContainer>
      <PredictionButton onClick={handlePrediction}>Get Prediction</PredictionButton>
      {prediction && (
        <PredictionResult>
          <h3>Next Day Prediction: {prediction.next_day_prediction}</h3>
          <h3>Last 10 Days Stock Data:</h3>
          <Table>
            <thead>
              <tr>
                <Th>Date</Th>
                <Th>Close</Th>
                <Th>SMA_10</Th>
                <Th>SMA_50</Th>
                <Th>RSI</Th>
              </tr>
            </thead>
            <tbody>
              {prediction.last_10_days.map((day, index) => (
                <tr key={index}>
                  <Td>{new Date(day.Date).toLocaleDateString()}</Td>
                  <Td>{day.Close}</Td>
                  <Td>{day.SMA_10}</Td>
                  <Td>{day.SMA_50}</Td>
                  <Td>{day.RSI}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={formatDataForChart(prediction)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="close" stroke="#8884d8" activeDot={{ r: 8 }} />
              <Line type="monotone" dataKey="next_day_prediction" stroke="#FF0000" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </PredictionResult>
      )}
    </PredictionContainer>
  );
};

export default PredictionModel;