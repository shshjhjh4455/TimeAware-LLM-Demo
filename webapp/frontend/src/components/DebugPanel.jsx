import React from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider,
  Table,
  TableBody,
  TableRow,
  TableCell,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExploreIcon from '@mui/icons-material/Explore';
import VerifiedIcon from '@mui/icons-material/Verified';
import StorageIcon from '@mui/icons-material/Storage';
import TimerIcon from '@mui/icons-material/Timer';
import MemoryIcon from '@mui/icons-material/Memory';
import TimelineIcon from '@mui/icons-material/Timeline';
import ArticleIcon from '@mui/icons-material/Article';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PlaceIcon from '@mui/icons-material/Place';

function InfoRow({ label, value, mono }) {
  return (
    <TableRow sx={{ '& td': { py: 0.3, px: 0.5, border: 0, fontSize: '0.8rem' } }}>
      <TableCell sx={{ color: 'text.secondary', whiteSpace: 'nowrap', width: 100 }}>
        {label}
      </TableCell>
      <TableCell sx={{ fontFamily: mono ? 'monospace' : 'inherit', wordBreak: 'break-all' }}>
        {value}
      </TableCell>
    </TableRow>
  );
}

export default function DebugPanel({ debug }) {
  if (!debug) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Send a message to see pipeline debug info.
        </Typography>
      </Box>
    );
  }

  const sys = debug.system || {};
  const nav = debug.navigator || {};
  const ver = debug.verifier || {};
  const temporal = debug.temporal || {};
  const knowledge = debug.knowledge || {};
  const prompt = debug.prompt || {};
  const gen = debug.generation || {};
  const charMap = debug.character_mapping || {};
  const qScene = debug.question_scene || {};

  return (
    <Box sx={{ p: 1.5 }}>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700, fontSize: '0.95rem' }}>
        Pipeline Debug
      </Typography>

      {/* Summary bar */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1.5 }}>
        <Chip icon={<TimerIcon />} label={`${debug.time_ms}ms`} size="small" variant="outlined" />
        <Chip
          label={temporal.is_future ? 'FUTURE' : 'PAST'}
          size="small"
          color={temporal.is_future ? 'warning' : 'success'}
        />
        <Chip icon={<StorageIcon />} label={`${knowledge.total_count || 0} knowledge`} size="small" variant="outlined" />
        {sys.gpu_available && (
          <Chip icon={<MemoryIcon />} label={sys.gpu_name || 'GPU'} size="small" color="info" variant="outlined" />
        )}
      </Box>

      {/* System Info */}
      <Accordion disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <MemoryIcon fontSize="small" sx={{ color: 'grey.500' }} />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>System</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Table size="small">
            <TableBody>
              <InfoRow label="Navigator" value={sys.navigator_type} mono />
              <InfoRow label="Chat Model" value={sys.chat_model} mono />
              <InfoRow label="Gen Model" value={sys.generation_model} mono />
              <InfoRow label="Top-k" value={sys.top_k_chapters} mono />
              <InfoRow label="Memory Top-k" value={sys.memory_top_k} mono />
              <InfoRow label="Knowledge Order" value={sys.knowledge_order} />
              <InfoRow label="GPU" value={sys.gpu_available ? sys.gpu_name : 'Not used'} />
            </TableBody>
          </Table>
        </AccordionDetails>
      </Accordion>

      <Divider sx={{ my: 0.5 }} />

      {/* Step 0-1: Navigator */}
      <Accordion defaultExpanded disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ExploreIcon fontSize="small" color="primary" />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>
              Step 0-1. Navigator
            </Typography>
            <Chip label={`${nav.hypotheses?.length || 0} hypotheses`} size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
            Model: RoBERTa-Large ({nav.model_type || 'full_ft'}) | Classes: {nav.num_classes || '?'} | Device: {nav.device || '?'}
          </Typography>
          {nav.hypotheses?.map((h, i) => (
            <Box
              key={i}
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                py: 0.4,
                px: 1,
                bgcolor: i === 0 ? 'action.selected' : 'transparent',
                borderRadius: 1,
                mb: 0.3,
                borderLeft: i === 0 ? 3 : 0,
                borderColor: 'primary.main',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography variant="caption" color="text.secondary">#{i + 1}</Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                  {h.chapter_id}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{
                  width: 60,
                  height: 4,
                  bgcolor: 'action.hover',
                  borderRadius: 2,
                  overflow: 'hidden',
                }}>
                  <Box sx={{
                    width: `${Math.min(h.confidence * 100, 100)}%`,
                    height: '100%',
                    bgcolor: h.confidence > 0.7 ? 'success.main' : h.confidence > 0.5 ? 'warning.main' : 'error.main',
                    borderRadius: 2,
                  }} />
                </Box>
                <Typography variant="caption" sx={{ fontFamily: 'monospace', minWidth: 42, textAlign: 'right' }}>
                  {(h.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>
            </Box>
          ))}
        </AccordionDetails>
      </Accordion>

      {/* Step 0-2: Verifier */}
      <Accordion defaultExpanded disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <VerifiedIcon fontSize="small" color="secondary" />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>
              Step 0-2. Verifier
            </Typography>
            {ver.premise_verified !== undefined && (
              <Chip
                label={ver.premise_verified ? 'Premise OK' : 'Premise FAIL'}
                size="small"
                color={ver.premise_verified ? 'success' : 'error'}
                sx={{ height: 18, fontSize: '0.7rem' }}
              />
            )}
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
            Confidence: {((ver.confidence || 0) * 100).toFixed(1)}% | Model: {sys.chat_model}
          </Typography>
          {ver.selected_scenes?.map((s, i) => (
            <Box key={i} sx={{ py: 0.4, px: 1, bgcolor: 'action.hover', borderRadius: 1, mb: 0.3 }}>
              <Typography variant="body2" sx={{ fontWeight: 500, fontSize: '0.8rem' }}>
                {s.scene_title || s.scene_id}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                {s.chapter_id} / {s.scene_id}
              </Typography>
            </Box>
          ))}
          {ver.reasoning && (
            <Box sx={{ mt: 0.5, p: 1, bgcolor: 'action.hover', borderRadius: 1 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>Reasoning:</Typography>
              <Typography variant="body2" sx={{ fontSize: '0.75rem', mt: 0.3 }}>
                {ver.reasoning}
              </Typography>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>

      {/* Step 1: Character Mapping */}
      <Accordion disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PlaceIcon fontSize="small" sx={{ color: '#66bb6a' }} />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>
              Step 1. Character Mapping
            </Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Table size="small">
            <TableBody>
              <InfoRow label="Input Period" value={charMap.input_period} />
              <InfoRow label="Mapped To" value={charMap.mapped_position} mono />
              <InfoRow label="Question At" value={temporal.question_period} mono />
            </TableBody>
          </Table>
        </AccordionDetails>
      </Accordion>

      {/* Step 3: Temporal */}
      <Accordion disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TimelineIcon fontSize="small" sx={{ color: temporal.is_future ? '#ff9800' : '#4caf50' }} />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>
              Step 3. Temporal Classification
            </Typography>
            <Chip
              label={temporal.is_future ? 'FUTURE' : 'PAST'}
              size="small"
              color={temporal.is_future ? 'warning' : 'success'}
              sx={{ height: 18, fontSize: '0.7rem' }}
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Box sx={{ px: 1, py: 0.5 }}>
            <Typography variant="body2" sx={{ fontSize: '0.8rem', fontFamily: 'monospace' }}>
              Character NOW: {temporal.current_position}
            </Typography>
            <Typography variant="body2" sx={{ fontSize: '0.8rem', fontFamily: 'monospace' }}>
              Question AT:{'  '}{temporal.question_period}
            </Typography>
            <Typography variant="body2" sx={{ fontSize: '0.8rem', mt: 0.5 }}>
              {temporal.is_future
                ? 'Question refers to an event AFTER the character\'s current time. Negative constraint applied.'
                : 'Question refers to an event BEFORE or AT the character\'s current time. Knowledge provided.'}
            </Typography>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Step 4: Knowledge */}
      <Accordion defaultExpanded disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <StorageIcon fontSize="small" color="info" />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>
              Step 4. Knowledge Retrieval
            </Typography>
            <Chip
              label={`${knowledge.total_count || 0} items`}
              size="small"
              variant="outlined"
              sx={{ height: 18, fontSize: '0.7rem' }}
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
            MemoryRetriever: {knowledge.memory_count || 0} memories (min_sim: 0.45)
          </Typography>

          {/* Type distribution chips */}
          {knowledge.type_distribution && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
              {Object.entries(knowledge.type_distribution).map(([type, count]) => (
                <Chip
                  key={type}
                  label={`${type}: ${count}`}
                  size="small"
                  variant="outlined"
                  color={
                    type === 'FACT_LEARNED' ? 'primary' :
                    type === 'BELIEF_CHANGED' ? 'secondary' :
                    type === 'RELATIONSHIP_UPDATED' ? 'warning' : 'default'
                  }
                  sx={{ height: 20, fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          )}

          {/* Individual knowledge items */}
          {knowledge.items?.map((item, i) => (
            <Box
              key={i}
              sx={{
                p: 1,
                mb: 0.5,
                bgcolor: 'action.hover',
                borderRadius: 1,
                borderLeft: 3,
                borderColor:
                  item.type === 'FACT_LEARNED' ? 'primary.main' :
                  item.type === 'BELIEF_CHANGED' ? 'secondary.main' :
                  item.type === 'RELATIONSHIP_UPDATED' ? 'warning.main' : 'grey.500',
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.3 }}>
                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                  {item.type}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', color: 'text.secondary', fontSize: '0.7rem' }}>
                    {item.chapter_id}
                  </Typography>
                  {item.similarity > 0 && (
                    <Chip
                      label={`sim: ${item.similarity.toFixed(3)}`}
                      size="small"
                      sx={{ height: 16, fontSize: '0.65rem' }}
                    />
                  )}
                </Box>
              </Box>
              <Typography variant="body2" sx={{ fontSize: '0.78rem', lineHeight: 1.4 }}>
                {item.description}
              </Typography>
            </Box>
          ))}
        </AccordionDetails>
      </Accordion>

      {/* Step 5-6: Generation */}
      <Accordion disableGutters elevation={0} sx={{ bgcolor: 'transparent', '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 36, px: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SmartToyIcon fontSize="small" sx={{ color: '#ab47bc' }} />
            <Typography variant="subtitle2" sx={{ fontSize: '0.85rem' }}>
              Step 5-6. Prompt + Generation
            </Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 0.5, py: 0 }}>
          <Table size="small">
            <TableBody>
              <InfoRow label="Prompt Length" value={`${(prompt.length_chars || 0).toLocaleString()} chars`} mono />
              <InfoRow label="Response Length" value={`${(gen.response_length || 0).toLocaleString()} chars`} mono />
              <InfoRow label="Model" value={sys.generation_model} mono />
              <InfoRow label="Has [RESPONSE]" value={gen.has_response_tag ? 'Yes' : 'No'} />
              <InfoRow label="Has Premise Check" value={gen.has_premise_check ? 'Yes' : 'No'} />
              <InfoRow label="Has Analysis" value={gen.has_analysis ? 'Yes' : 'No'} />
              <InfoRow label="Future Knowledge" value={prompt.future_knowledge_items || 0} mono />
            </TableBody>
          </Table>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
