import React from 'react';
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Avatar,
  Box,
  Typography,
} from '@mui/material';

const AVATAR_COLORS = {
  HP: '#c62828',
  HG: '#1565c0',
  RW: '#e65100',
};

export default function CharacterSelector({
  characters,
  selectedCharacter,
  selectedPeriod,
  onCharacterChange,
  onPeriodChange,
}) {
  const currentChar = characters.find((c) => c.name === selectedCharacter);

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      {currentChar && (
        <Avatar
          sx={{
            bgcolor: AVATAR_COLORS[currentChar.avatar] || 'primary.main',
            width: 40,
            height: 40,
            fontWeight: 600,
            fontSize: '0.85rem',
          }}
        >
          {currentChar.avatar}
        </Avatar>
      )}

      <FormControl size="small" sx={{ minWidth: 180 }}>
        <InputLabel>Character</InputLabel>
        <Select
          value={selectedCharacter}
          label="Character"
          onChange={(e) => onCharacterChange(e.target.value)}
        >
          {characters.map((c) => (
            <MenuItem key={c.name} value={c.name}>
              <Typography variant="body2">{c.name}</Typography>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl size="small" sx={{ minWidth: 320 }}>
        <InputLabel>Time Period</InputLabel>
        <Select
          value={selectedPeriod}
          label="Time Period"
          onChange={(e) => onPeriodChange(e.target.value)}
        >
          {currentChar?.periods.map((p) => (
            <MenuItem key={p.value} value={p.value}>
              <Typography variant="body2">{p.label}</Typography>
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );
}
