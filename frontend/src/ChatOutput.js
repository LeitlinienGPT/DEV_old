import React from 'react';
import './ChatOutput.css';
import Card from '@mui/joy/Card';
import CardContent from '@mui/joy/CardContent';
import Typography from '@mui/joy/Typography';
import Skeleton from '@mui/joy/Skeleton'; // Import Skeleton

const ChatOutput = ({ messages, isLoading }) => { // Add isLoading prop
  const lastMessage = messages[messages.length - 1] || {};
  const question = lastMessage.question || "Bitte stelle eine Frage, um eine Antwort zu erhalten.";

  // Ausgabe in der Browser-Konsole
  // console.log("Alle Nachrichten:", messages);
  // console.log("Letzte Nachricht:", lastMessage);
  // console.log("Antwort:", answer);
  // console.log("Quellendokumente:", sourceDocuments);


  return (
    <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap', paddingTop: '1rem', paddingBottom: '1rem' }}>
      {/* Fragekarte */}
      <Card sx={{ maxWidth: 900, width: '95%', borderRadius: '16px', boxShadow: 3}}>
        <CardContent>
          <Typography level="title-md" sx={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Frage</Typography>
          <Typography level="body2" color="text.secondary" component="div" sx={{ textAlign: 'justify' }}>
            {question}
          </Typography>
        </CardContent>
      </Card>

      {/* Antwortkarte with Skeleton */}
      <Card sx={{ maxWidth: 900, width: '95%', borderRadius: '16px', boxShadow: 3}}>
        <CardContent>
          <Typography level="title-md" sx={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Antwort</Typography>
          {isLoading ? (
            <div>
              <Skeleton variant="text" width="80%" />
              <Skeleton variant="rectangular" height="2rem" />
              <Skeleton variant="text" width="90%" />
            </div>
          ) : (
            <Typography level="body2" color="text.secondary" component="div" sx={{ textAlign: 'justify' }}>
              {lastMessage.answer || "Warte auf die Antwort..."}
            </Typography>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ChatOutput;
