import React from 'react';
import styled from 'styled-components';

const Card = styled.div`
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  margin: 10px;
  text-align: center;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
`;

const Symbol = styled.h3`
  font-size: 18px;
  margin: 10px 0;
`;

const Price = styled.p`
  font-size: 16px;
  color: #333;
`;

const Volume = styled.p`
  font-size: 14px;
  color: #777;
`;

const StockCard = ({ symbol, price, volume }) => (
  <Card>
    <Symbol>{symbol}</Symbol>
    <Price>Price: ${price}</Price>
    <Volume>Volume: {volume}</Volume>
  </Card>
);

export default StockCard;