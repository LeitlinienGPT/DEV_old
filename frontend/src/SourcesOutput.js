import React, { useState } from 'react';
import './ChatOutput.css';
import Card from '@mui/joy/Card';
import CardContent from '@mui/joy/CardContent';
import Typography from '@mui/joy/Typography';
import Link from '@mui/joy/Link';
import Button from '@mui/joy/Button';
import Skeleton from '@mui/joy/Skeleton'; // Import Skeleton

const SourcesOutput = ({ sourceDocuments, isLoading }) => { 
  const [showContentStates, setShowContentStates] = useState(Array(sourceDocuments.length).fill(false));
  
  // Print raw data for debugging
  console.log("Raw sourceDocuments:", sourceDocuments); 
  console.log("Type of sourceDocuments:", typeof sourceDocuments); 

  const formatTitleAndExtractRegisterNumber = (title) => {
    console.log("Title:", title)
    const parts = title.split('_');
    console.log("title parts:", parts)
    let registerNumber = parts[0];
    if (registerNumber && registerNumber.length >= 6) {
      registerNumber = `${registerNumber.substring(0, 3)}-${registerNumber.substring(4, 7)}`;
    } else {
      registerNumber = "Unbekannt";
    }
    const formattedTitle = parts.slice(2).join(', ').replace(/-/g, ' ');
    return { formattedTitle, registerNumber };
  };

  const renderValidityButton = (validity) => {
    return (
      <Button variant="soft" color={validity === "Gültig" ? "success" : "danger"}>
        {validity}
      </Button>
    );
  };

  const toggleShowContent = (index) => {
    const updatedShowContentStates = [...showContentStates];
    updatedShowContentStates[index] = !updatedShowContentStates[index];
    setShowContentStates(updatedShowContentStates);
  };

  const renderSourceDocumentSkeleton = () => {
    return (
      <div style={{ marginBottom: '1rem' }}>
        <Skeleton variant="text" width="60%" />
        <Skeleton variant="rectangular" height="2rem" />
        <Skeleton variant="text" width="80%" />
        <Skeleton variant="rectangular" height="2rem" />
        <Skeleton variant="text" width="70%" />
        <Skeleton variant="rectangular" height="2rem" />
        <Skeleton variant="text" width="90%" />
        <Skeleton variant="rectangular" height="2rem" />
        <Skeleton variant="text" width="50%" />
        <Skeleton variant="rectangular" height="2rem" />
        <hr />
      </div>
    );
  };

  const renderSourceDocuments = () => {
    return sourceDocuments.map((doc, index) => {
      const { formattedTitle, registerNumber } = formatTitleAndExtractRegisterNumber(doc.metadata.Source.split('/').pop().replace('.pdf', '')); 
      const awmfRegisterUrl = registerNumber !== "Unbekannt" 
        ? `https://register.awmf.org/de/leitlinien/detail/${registerNumber}`
        : null;

      return (
        <div key={index} style={{ marginBottom: '1rem' }}>
          {/*  Gültigkeit is now removed */}
          {doc.metadata.Registernummer && ( // Only render if Registernummer exists
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
              <Typography fontWeight="bold" component="span" style={{ marginRight: '0.5rem' }}>Registernummer:</Typography>
              {awmfRegisterUrl 
                ? <Link href={awmfRegisterUrl} target="_blank" variant="outlined">
                    {doc.metadata.Registernummer}
                  </Link>
                : <span>{doc.metadata.Registernummer}</span>
              }
            </div>
          )}
          {doc.metadata.Datenbank && ( // Only render if Datenbank exists
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
              <Typography fontWeight="bold" component="span" style={{ marginRight: '0.5rem' }}>Datenbank:</Typography>
              <span>{doc.metadata.Datenbank}</span>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
            <Typography fontWeight="bold" component="span" style={{ marginRight: '0.5rem' }}>Titel:</Typography>
            <span>{doc.metadata.Source}</span> {/* Use source directly as the title */}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
            <Typography fontWeight="bold" component="span" style={{ marginRight: '0.5rem' }}>Seite (im PDF):</Typography>
            <span>{doc.metadata.Page}</span> {/* Use Page directly */}
          </div>
          {/*  Content section removed for now */}
          {/* <div style={{ display: 'flex', alignItems: 'center', marginBottom: '0.5rem' }}>
            <Typography fontWeight="bold" component="span" style={{ marginRight: '0.5rem' }}>Inhalt:</Typography>
            <Button onClick={() => toggleShowContent(index)} variant="outlined" size="small">
              Mehr Informationen
            </Button>
          </div>
          {showContentStates[index] && <div style={{ marginTop: '0.5rem' }}>{doc.page_content}</div>} */}
          <hr />
        </div>
      );
    });
  };

  return (
    <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap', paddingTop: '1rem', paddingBottom: '1rem' }}>
      <Card sx={{ maxWidth: 1200, width: '95%', borderRadius: '16px', boxShadow: 3, bgcolor: 'primary.box' }}>
        <CardContent>
          <Typography level="title-md" sx={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Quellen</Typography>
          {isLoading ? (
            <>
              {renderSourceDocumentSkeleton()}
              {renderSourceDocumentSkeleton()}
            </>
          ) : (
            <Typography variant="body2" color="text.secondary" component="div" sx={{ textAlign: 'justify' }}>
              {sourceDocuments.length > 0 ? renderSourceDocuments() : "Keine Quellen gefunden"}
            </Typography>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default SourcesOutput;