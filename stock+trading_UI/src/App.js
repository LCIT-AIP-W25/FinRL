import React from 'react';
import styled from 'styled-components';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle, faUsers, faEnvelope } from '@fortawesome/free-solid-svg-icons';
import Home from './components/Home.js';
import TeamMemberCard from './components/TeamMemberCard.js';

const Container = styled.div`
  font-family: Arial, sans-serif;
`;

const Header = styled.header`
  background: #4CAF50;
  color: white;
  padding: 10px 20px;
  text-align: center;
  position: relative;
`;

const Nav = styled.nav`
  display: flex;
  justify-content: center;
  margin: 20px 0;

  a {
    margin: 0 15px;
    text-decoration: none;
    color: white;
    font-weight: bold;
    display: flex;
    align-items: center;

    &:hover {
      text-decoration: underline;
    }

    svg {
      margin-right: 8px;
    }
  }
`;

const Section = styled.section`
  padding: 20px;
  text-align: center;
`;

const TeamGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
`;

const LogoContainer = styled.div`
  position: absolute;
  top: 10px;
  left: 10px;
  display: flex;
  align-items: center;
`;

const Logo = styled.img`
  width: 50px;
  height: 50px;
  margin-right: 10px;
`;

const TeamName = styled.h1`
  font-size: 24px;
  font-weight: bold;
  color: white;
`;

const App = () => {
  const teamMembers = [
    { image: 'path/to/image1.jpg', name: 'Navneet Kaur', designation: 'Team Leader' },
    { image: 'path/to/image2.jpg', name: 'Jashandeep Singh', designation: 'Data Analyst' },
    { image: 'path/to/image3.jpg', name: 'Jini Zacharias', designation: 'Technical Lead' },
    { image: 'path/to/image4.jpg', name: 'Yatin Goyal', designation: 'Quality Assurance' },
    { image: 'path/to/image5.jpg', name: 'Ardra Nair', designation: 'Software Engineer' },
    { image: 'path/to/image6.jpg', name: 'Atif Ahmed', designation: 'Project Coordinator' },
  ];

  return (
    <Router>
      <Container>
        <Header>
          <LogoContainer>
            <Logo src="C:/Users/aswin/OneDrive/Desktop/stockYTradingbotUI/stock-trading-bot/public/Equitrade4.png" alt="Team Logo" />
            <TeamName>Team Name</TeamName>
          </LogoContainer>
          <h1>EQUITRADE</h1>
          <Nav>
            <Link to="/">
              <FontAwesomeIcon icon={faHome} />
              Home
            </Link>
            <Link to="/about">
              <FontAwesomeIcon icon={faInfoCircle} />
              About
            </Link>
            <Link to="/team">
              <FontAwesomeIcon icon={faUsers} />
              Team
            </Link>
            <Link to="/contact">
              <FontAwesomeIcon icon={faEnvelope} />
              Contact
            </Link>
          </Nav>
        </Header>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={
            <Section id="about">
              <h2>About Us</h2>
              <p>Welcome to EQUITRADE, your trusted partner in stock trading.</p>
            </Section>
          } />
          <Route path="/team" element={
            <Section id="team">
              <h2>Our Team</h2>
              <TeamGrid>
                {teamMembers.map((member, index) => (
                  <TeamMemberCard
                    key={index}
                    image={member.image}
                    name={member.name}
                    designation={member.designation}
                  />
                ))}
              </TeamGrid>
            </Section>
          } />
          <Route path="/contact" element={
            <Section id="contact">
              <h2>Contact Us</h2>
              <p>Get in touch with us for any inquiries.</p>
            </Section>
          } />
        </Routes>
      </Container>
    </Router>
  );
};

export default App;