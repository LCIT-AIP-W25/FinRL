import React from 'react';
import styled from 'styled-components';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle, faUsers, faEnvelope, faBullseye, faEye, faHandshake, faChartLine, faCogs, faLightbulb, faUserFriends, faStar, faPhone, faMapMarkerAlt } from '@fortawesome/free-solid-svg-icons';
import Home from './components/Home';
import TeamMemberCard from './components/TeamMemberCard';
import ChatbotPage from './components/ChatbotPage';
import PredictionModel from './components/PredictionModel'; // Import the new PredictionModel component

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

const AboutSection = styled(Section)`
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  max-width: 800px;
  width: 100%;
  text-align: left;
  margin: 20px auto;
`;

const ContactSection = styled(Section)`
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  max-width: 800px;
  width: 100%;
  text-align: left;
  margin: 20px auto;
`;

const Icon = styled(FontAwesomeIcon)`
  margin-right: 10px;
  color: #4CAF50;
`;

const List = styled.ul`
  list-style: none;
  padding: 0;
`;

const ListItem = styled.li`
  margin: 10px 0;
  display: flex;
  align-items: center;
`;

const ContactInfo = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-top: 20px;
`;

const ContactItem = styled.div`
  margin: 10px 0;
  display: flex;
  align-items: center;
`;

const App = () => {
  const teamMembers = [
    { image: './navneet.jpeg', name: 'Navneet Kaur', designation: 'Team Leader' },
    { image: './jashan.jpeg', name: 'Jashandeep Singh', designation: 'Data Analyst' },
    { image: './jini.jpeg', name: 'Jini Zacharias', designation: 'Technical Lead' },
    { image: './Yatin.jpeg', name: 'Yatin Goyal', designation: 'Quality Assurance' },
    { image: './ardra.jpeg', name: 'Ardra Nair', designation: 'Software Engineer' },
    { image: './atif.jpeg', name: 'Atif Ahmed', designation: 'Project Coordinator' },
  ];

  return (
    <Router>
      <Container>
        <Header>
          <LogoContainer>
            <Link to="/predict"> {/* Add link to prediction model */}
              <Logo src="./Equitrade4.png" alt="Team Logo" />
              <TeamName>EQUITRADE</TeamName>
            </Link>
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
            <AboutSection id="about">
              <h2>About Us</h2>
              <p>Welcome to EQUITRADE, your trusted partner in stock trading.</p>
              <h3><Icon icon={faBullseye} /> Our Mission</h3>
              <p>To empower investors with cutting-edge tools and insights to make informed trading decisions.</p>
              <h3><Icon icon={faEye} /> Our Vision</h3>
              <p>To be the leading platform for stock trading, recognized for innovation, reliability, and customer satisfaction.</p>
              <h3><Icon icon={faHandshake} /> Our Values</h3>
              <List>
                <ListItem><Icon icon={faCogs} /> <strong>Integrity:</strong> We uphold the highest standards of integrity in all our actions.</ListItem>
                <ListItem><Icon icon={faLightbulb} /> <strong>Innovation:</strong> We constantly seek innovative solutions to meet our clients' needs.</ListItem>
                <ListItem><Icon icon={faUserFriends} /> <strong>Customer Focus:</strong> We prioritize our customers and strive to exceed their expectations.</ListItem>
                <ListItem><Icon icon={faStar} /> <strong>Excellence:</strong> We are committed to excellence in everything we do.</ListItem>
              </List>
              <h3><Icon icon={faChartLine} /> Our Services</h3>
              <List>
                <ListItem><Icon icon={faChartLine} /> <strong>Real-Time Stock Data:</strong> Access to live stock prices and market data.</ListItem>
                <ListItem><Icon icon={faChartLine} /> <strong>Advanced Analytics:</strong> Comprehensive tools for technical and fundamental analysis.</ListItem>
                <ListItem><Icon icon={faChartLine} /> <strong>Trading Insights:</strong> Expert insights and recommendations to guide your trading decisions.</ListItem>
                <ListItem><Icon icon={faChartLine} /> <strong>Portfolio Management:</strong> Tools to manage and optimize your investment portfolio.</ListItem>
              </List>
            </AboutSection>
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
            <ContactSection id="contact">
              <h2>Contact Us</h2>
              <p>Get in touch with us for any inquiries.</p>
              <ContactInfo>
                <ContactItem>
                  <Icon icon={faPhone} /> <span>Phone: +1 (123) 456-7890</span>
                </ContactItem>
                <ContactItem>
                  <Icon icon={faEnvelope} /> <span>Email: support@equitrade.com</span>
                </ContactItem>
                <ContactItem>
                  <Icon icon={faMapMarkerAlt} /> <span>Address: 1234 Market St, San Francisco, CA 94103</span>
                </ContactItem>
              </ContactInfo>
            </ContactSection>
          } />
          <Route path="/predict" element={<PredictionModel />} /> {/* Add the new route */}
        </Routes>
      </Container>
    </Router>
  );
};

export default App;
