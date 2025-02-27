import React from 'react';
import styled from 'styled-components';
import { animated } from '@react-spring/web';

const Card = styled(animated.div)`
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  margin: 10px;
  padding: 10px;
  width: 150px;
  text-align: center;
  position: relative;
`;


const Symbol = styled.h2`
  color: #4CAF50;
  font-size: 1.2em;
`;

const Price = styled.p`
  color: #888;
  font-size: 1.2em;
  margin: 5px 0;
`;

const Volume = styled.p`
  color: #888;
  font-size: 0.9em;
`;

const StockCard = ({ symbol, price, volume, style }) => {
  return (
    <Card style={style}>
      <Symbol>{symbol}</Symbol>
      <Price>Price: ${price}</Price>
      <Volume>Volume: {volume}</Volume>
    </Card>
  );
};

export default StockCard;