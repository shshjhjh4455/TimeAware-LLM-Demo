import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, Paper, Typography, CircularProgress, IconButton } from '@mui/material';
import BugReportIcon from '@mui/icons-material/BugReport';
import CharacterSelector from './CharacterSelector';
import ChatMessage from './ChatMessage';
import InputBox from './InputBox';
import DebugPanel from './DebugPanel';
import { fetchCharacters, sendChatStream, fetchHealth } from '../services/api';

export default function Chat() {
  const [characters, setCharacters] = useState([]);
  const [selectedCharacter, setSelectedCharacter] = useState('');
  const [selectedPeriod, setSelectedPeriod] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pipelineReady, setPipelineReady] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [activeDebug, setActiveDebug] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Check pipeline health and load characters
  useEffect(() => {
    const init = async () => {
      try {
        const health = await fetchHealth();
        setPipelineReady(health.pipeline_ready);
        const chars = await fetchCharacters();
        setCharacters(chars);
        if (chars.length > 0) {
          setSelectedCharacter(chars[0].name);
          if (chars[0].periods.length > 0) {
            setSelectedPeriod(chars[0].periods[0].value);
          }
        }
      } catch (err) {
        console.error('Init error:', err);
      }
    };
    init();

    // Poll health until ready
    const interval = setInterval(async () => {
      try {
        const health = await fetchHealth();
        if (health.pipeline_ready) {
          setPipelineReady(true);
          clearInterval(interval);
        }
      } catch {
        // ignore
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const handleSend = async (question) => {
    if (!question.trim() || loading || !pipelineReady) return;

    const userMsg = { role: 'user', content: question, msgId: `user-${Date.now()}`, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    // Unique ID to identify the AI placeholder message (avoids timestamp collision)
    const aiMsgId = `ai-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const placeholderMsg = {
      role: 'assistant',
      content: '',
      character: selectedCharacter,
      character_period: selectedPeriod,
      debug: null,
      msgId: aiMsgId,
      timestamp: Date.now(),
      isStreaming: true,
    };
    setMessages((prev) => [...prev, placeholderMsg]);

    try {
      await sendChatStream(
        {
          character: selectedCharacter,
          character_period: selectedPeriod,
          question,
        },
        {
          onDebug: (event) => {
            setActiveDebug(event.debug);
            setMessages((prev) =>
              prev.map((m) =>
                m.msgId === aiMsgId
                  ? { ...m, debug: event.debug, character: event.character, character_period: event.character_period }
                  : m
              )
            );
          },
          onToken: (content) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.msgId === aiMsgId
                  ? { ...m, content: m.content + content }
                  : m
              )
            );
          },
          onDone: (event) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.msgId === aiMsgId
                  ? { ...m, content: event.answer, debug: event.debug, isStreaming: false }
                  : m
              )
            );
            setActiveDebug(event.debug);
            setLoading(false);
          },
          onError: (err) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.msgId === aiMsgId
                  ? { ...m, content: `Error: ${err.message}`, isError: true, isStreaming: false }
                  : m
              )
            );
            setLoading(false);
          },
        }
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.msgId === aiMsgId
            ? { ...m, content: `Error: ${err.message}`, isError: true, isStreaming: false }
            : m
        )
      );
      setLoading(false);
    }
  };

  const currentChar = characters.find((c) => c.name === selectedCharacter);

  return (
    <Box sx={{ display: 'flex', height: '100%' }}>
      {/* Main chat area */}
      <Box
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
        }}
      >
        {/* Character selector bar */}
        <Paper
          elevation={0}
          sx={{
            p: 2,
            borderBottom: 1,
            borderColor: 'divider',
            borderRadius: 0,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <CharacterSelector
            characters={characters}
            selectedCharacter={selectedCharacter}
            selectedPeriod={selectedPeriod}
            onCharacterChange={(name) => {
              setSelectedCharacter(name);
              const char = characters.find((c) => c.name === name);
              if (char && char.periods.length > 0) {
                setSelectedPeriod(char.periods[0].value);
              }
            }}
            onPeriodChange={setSelectedPeriod}
          />
          <Box sx={{ flexGrow: 1 }} />
          <IconButton
            onClick={() => setShowDebug(!showDebug)}
            color={showDebug ? 'primary' : 'default'}
            title="Toggle debug panel"
          >
            <BugReportIcon />
          </IconButton>
        </Paper>

        {/* Pipeline status */}
        {!pipelineReady && (
          <Box
            sx={{
              p: 2,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              bgcolor: 'action.hover',
            }}
          >
            <CircularProgress size={16} />
            <Typography variant="body2" color="text.secondary">
              Initializing pipeline... This may take a few seconds.
            </Typography>
          </Box>
        )}

        {/* Messages area */}
        <Box
          sx={{
            flexGrow: 1,
            overflow: 'auto',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {messages.length === 0 && pipelineReady && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                opacity: 0.5,
              }}
            >
              <Typography variant="h5" gutterBottom>
                Start a conversation
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Select a character and time period, then ask a question.
              </Typography>
            </Box>
          )}
          {messages.map((msg, idx) => (
            <ChatMessage
              key={idx}
              message={msg}
              avatar={currentChar?.avatar}
              onDebugClick={
                msg.debug ? () => { setActiveDebug(msg.debug); setShowDebug(true); } : undefined
              }
            />
          ))}
          {loading && !messages.some((m) => m.isStreaming && m.content) && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, pl: 6 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                {selectedCharacter} is thinking...
              </Typography>
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Box>

        {/* Input box */}
        <InputBox onSend={handleSend} disabled={loading || !pipelineReady} />
      </Box>

      {/* Debug panel */}
      {showDebug && (
        <Box
          sx={{
            width: 520,
            minWidth: 420,
            borderLeft: 1,
            borderColor: 'divider',
            overflow: 'auto',
          }}
        >
          <DebugPanel debug={activeDebug} />
        </Box>
      )}
    </Box>
  );
}
