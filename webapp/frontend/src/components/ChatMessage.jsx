import React from 'react';
import { Box, Paper, Avatar, Typography, IconButton } from '@mui/material';
import BugReportIcon from '@mui/icons-material/BugReport';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const AVATAR_COLORS = {
  HP: '#c62828',
  HG: '#1565c0',
  RW: '#e65100',
};

export default function ChatMessage({ message, avatar, onDebugClick }) {
  const isUser = message.role === 'user';
  const isError = message.isError;

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        gap: 1.5,
        maxWidth: '85%',
        alignSelf: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      {!isUser && (
        <Avatar
          sx={{
            bgcolor: isError
              ? 'error.main'
              : AVATAR_COLORS[avatar] || 'primary.main',
            width: 36,
            height: 36,
            fontSize: '0.75rem',
            fontWeight: 600,
            mt: 0.5,
          }}
        >
          {avatar || 'AI'}
        </Avatar>
      )}

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        {!isUser && message.character && (
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ pl: 1 }}
          >
            {message.character} ({message.character_period})
          </Typography>
        )}

        <Paper
          elevation={0}
          sx={{
            p: 2,
            bgcolor: isUser
              ? 'primary.main'
              : isError
                ? 'error.dark'
                : 'background.paper',
            color: isUser ? 'primary.contrastText' : 'text.primary',
            border: isUser ? 'none' : 1,
            borderColor: 'divider',
            '& p': { m: 0 },
            '& p + p': { mt: 1 },
          }}
        >
          {isUser ? (
            <Typography variant="body1">{message.content}</Typography>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          )}
        </Paper>

        {!isUser && message.debug && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, pl: 1 }}>
            <IconButton size="small" onClick={onDebugClick} title="View debug info">
              <BugReportIcon fontSize="small" />
            </IconButton>
            <Typography variant="caption" color="text.secondary">
              {message.debug.time_ms}ms
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
}
