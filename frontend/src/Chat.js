import React, { useState } from 'react';
import './Chat.css';
import Box from '@mui/joy/Box';
import Button from '@mui/joy/Button';
import Textarea from '@mui/joy/Textarea';
import Select from '@mui/joy/Select';
import Option from '@mui/joy/Option';
import CircularProgress from '@mui/material/CircularProgress'; // Import CircularProgress for loading indicator

const Chat = ({ addMessage, setMessages, messages }) => {
  const [input, setInput] = useState('');
  const [dropdownValue, setDropdownValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false); // State to track if a request is being processed

  const handleSubmit = async (event) => {
    event.preventDefault();
    console.log('Input:', input.trim());
  
    // Log the environment variable
    console.log('Backend URL from env:', process.env.REACT_APP_BACKEND_URL);
  
    // Construct the URL and log it
    console.log('Backend URL from env:', process.env.REACT_APP_BACKEND_URL);
    const requestUrl = `${process.env.REACT_APP_BACKEND_URL}/process`;
    console.log('Request URL:', requestUrl);
    console.log('Environment variable:', process.env.REACT_APP_BACKEND_URL);
  
    if (input.trim()) {
      setIsSubmitting(true);
      try {
        // Log before sending the request
        console.log('Sending request to:', requestUrl);
  
        const response = await fetch(requestUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question: input.trim() }),
        });
  
        // Log the HTTP response status
        console.log('Response status:', response.status);
  
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
  
        const data = await response.json();
  
        // Log the received data
        console.log('Received data:', data);
  
        addMessage({ text: input.trim(), ...data });
  
        setInput('');
      } catch (error) {
        console.error('Error sending the query to the backend:', error);
      } finally {
        setIsSubmitting(false);
      }
    }
  };
  const handleClearChat = async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/clear_history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      // This response is just to confirm the action, you might not need to do anything with it
      const data = await response.json();
      console.log(data.message); // "History cleared"
  
      setMessages([]); // Assuming setMessages is the function from useState in your parent component
    } catch (error) {
      console.error('Error clearing the chat history:', error);
    }
  };

  return (
    <Box className="chat"> 
      <div className="chat-header">
        <h2>LeitlinienGPT</h2>
      </div>
      
      <Select
        value={dropdownValue}
        onChange={(event, value) => setDropdownValue(value)}
        placeholder="Wähle eine Option aus"
        size="md"
        sx={{
          marginY: 2,
          width: '100%',
          color: 'black', // Ensuring text is dark for better contrast
        }}
      >
        <Option value="Alle AMWF Leitlinien">Alle AMWF Leitlinien</Option>
        <Option value="Nur aktuell gültige Leitlinien">Nur aktuell gültige Leitlinien</Option>
      </Select>
      
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'space-between', variant: "soft"}}>
        <Button variant="solid" sx={{ marginY: 2 }} onClick={handleClearChat}>Chat Leeren</Button>
      </Box>  

      <Box component="form" className="chat-input-container" onSubmit={handleSubmit} sx={{ display: 'flex', gap: 2 }}>
        <Textarea
          size="lg"
          placeholder="Schreibe eine Nachricht..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          variant="soft"
          minRows={2}
          maxRows={6}
          sx={{
            flexGrow: 1,
            resize: 'none',
            width: '100%',
            color: 'black',
          }}
        />
        <Button type="submit" variant="solid" size="md" disabled={isSubmitting}>
          {isSubmitting ? <CircularProgress size={24} /> : 'Senden'}
        </Button>
      </Box>
    </Box>
  );
};

export default Chat;
