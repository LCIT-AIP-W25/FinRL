import React from 'react';
import styled from 'styled-components';

const Card = styled.div`
  margin: 20px;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  text-align: center;
  width: 400px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
`;

const Image = styled.img`
  width: 250px; /* Set a fixed width */
  height: 250px; /* Set a fixed height */
  border-radius: 50%;
  margin-bottom: 10px;
  object-fit: cover; /* Ensure the image covers the area without distortion */
  filter: contrast(1.2) brightness(1.1); /* Enhance image clarity */
`;

const Name = styled.h3`
  font-size: 18px;
  margin: 10px 0;
`;

const Designation = styled.p`
  font-size: 14px;
  color: #777;
`;

const TeamMemberCard = ({ image, name, designation }) => (
  <Card>
    <Image src={image} alt={name} />
    <Name>{name}</Name>
    <Designation>{designation}</Designation>
  </Card>
);

export default TeamMemberCard;