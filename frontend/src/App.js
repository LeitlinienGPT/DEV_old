import React, { useState } from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import Chat from './Chat';
import ChatOutput from './ChatOutput';
import SourcesOutput from './SourcesOutput';
import './App.css';
import { ThemeProvider } from '@mui/joy';
import joyTheme from './joyTheme'; // Ensure this is the correct path to your theme file

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const addMessage = (message) => {
    setIsLoading(true);
    // Simulate a delay to fetch data (replace this with your actual data fetching logic)
    setTimeout(() => {
      setMessages([...messages, message]);
      setIsLoading(false); // Set loading to false once data is fetched
    }, 2000); // Simulating a 2-second delay
  };

  return (
    <ThemeProvider theme={joyTheme}> {/* Wrap your app with ThemeProvider */}
      <CssBaseline />
      <div className="app-container" style={{ backgroundColor: joyTheme.colorSchemes.light.palette.primary.background }}>
        <div className="chat-layout">
          <div className="left-side">
            <Chat addMessage={addMessage} setMessages={setMessages} messages={messages}/>
            <ChatOutput messages={messages} isLoading={isLoading} /> {/* Pass isLoading to ChatOutput */}
          </div>
          <div className="right-side">
            <SourcesOutput sourceDocuments={messages.map(msg => msg.source_documents).flat()} isLoading={isLoading} />
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;
