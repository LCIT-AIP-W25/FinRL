import React, { useEffect, useState } from 'react';
import axios from 'axios';
import StockCard from './StockCard.js';
import { useTransition, animated } from '@react-spring/web';

const StockTable = () => {
  const [stocks, setStocks] = useState({});
  const [index, setIndex] = useState(0);

  useEffect(() => {
    axios.get('http://localhost:5000/stocks')
      .then(response => {
        console.log('Fetched data:', response.data); // Debugging log
        setStocks(response.data);
      })
      .catch(error => {
        console.error('There was an error fetching the stock data!', error);
      });
  }, []);

  useEffect(() => {
    if (Object.keys(stocks).length > 0) {
      const interval = setInterval(() => {
        setIndex(prevIndex => (prevIndex + 1) % Object.keys(stocks).length);
      }, 3000); // Change stock every 3 seconds
      return () => clearInterval(interval);
    }
  }, [stocks]);

  const stockSymbols = Object.keys(stocks);
  const transitions = useTransition(index, {
    from: { opacity: 0, transform: 'translate3d(0,100%,0)' },
    enter: { opacity: 1, transform: 'translate3d(0,0,0)' },
    leave: { opacity: 0, transform: 'translate3d(0,-50%,0)' },
    config: { tension: 200, friction: 20 },
  });

  return (
    <>
      {transitions((style, i) => {
        const symbols = [
          stockSymbols[(i + 0) % stockSymbols.length],
          stockSymbols[(i + 1) % stockSymbols.length],
          stockSymbols[(i + 2) % stockSymbols.length],
        ];
        return symbols.map(symbol => {
          const stock = stocks[symbol];
          if (!stock) return null; // Ensure stock data is available
          return (
            <animated.div style={style} key={symbol}>
              <StockCard
                symbol={symbol}
                price={stock.price}
                volume={stock.volume}
              />
            </animated.div>
          );
        });
      })}
    </>
  );
};

export default StockTable;