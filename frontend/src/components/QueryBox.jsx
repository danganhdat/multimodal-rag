import React, { useState } from "react";

import TextFormatRoundedIcon from "@mui/icons-material/TextFormatRounded";
import DrawRoundedIcon from "@mui/icons-material/DrawRounded";
import ImageSearchRoundedIcon from "@mui/icons-material/ImageSearchRounded";
import ReplayIcon from "@mui/icons-material/Replay";
import BorderColorIcon from "@mui/icons-material/BorderColor";
import SearchIcon from "@mui/icons-material/Search";
import {
  Box,
  Button,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from "@mui/material";

export default function QueryBox() {
  // state query by text, draw, image
  const [queryOption, setQueryOption] = useState("query-by-text");

  const handleQueryOptionChange = (event, newAlignment) => {
    if (newAlignment !== null) {
      setQueryOption(newAlignment);
    }
  };

  const [value_k, setValueK] = useState(10);
  const handleValueKChange = (event) => {
    setValueK(event.target.value);
  };

  const [valueFrame, setValueFrame] = useState(10);
  const handleValueFrameChange = (event) => {
    setValueFrame(event.target.value);
  };

  return (
    <Box
      sx={{
        backgroundColor: "white",
        borderRadius: "8px !important",
        padding: "10px !important",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100% !important",
      }}
    >
      {/* option input to query */}
      <Box
        sx={{
          width: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <ToggleButtonGroup
          value={queryOption}
          exclusive
          onChange={handleQueryOptionChange}
          aria-label="text alignment"
        >
          <ToggleButton
            value="query-by-text"
            aria-label="left aligned"
            sx={{
              backgroundColor:
                queryOption === "query-by-text"
                  ? "#0984e3 !important"
                  : "transparent",
            }}
          >
            <TextFormatRoundedIcon
              sx={{
                color: queryOption === "query-by-text" ? "white" : "black",
              }}
            />
          </ToggleButton>
          <ToggleButton
            value="query-by-draw"
            aria-label="centered"
            sx={{
              backgroundColor:
                queryOption === "query-by-draw"
                  ? "#0984e3 !important"
                  : "transparent",
            }}
          >
            <DrawRoundedIcon
              sx={{
                color: queryOption === "query-by-draw" ? "white" : "black",
              }}
            />
          </ToggleButton>
          <ToggleButton
            value="query-by-image"
            aria-label="right aligned"
            sx={{
              backgroundColor:
                queryOption === "query-by-image"
                  ? "#0984e3 !important"
                  : "transparent",
            }}
          >
            <ImageSearchRoundedIcon
              sx={{
                color: queryOption === "query-by-image" ? "white" : "black",
              }}
            />
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* query input */}
      <Box sx={{ width: "100%", marginTop: "10px" }}>
        <TextField
          placeholder="Enter your query"
          fullWidth
          multiline
          rows={4}
          id="fullWidth"
          sx={{
            "& .MuiInputBase-root": {
              minHeight: "100px",
            },
          }}
        />
      </Box>

      {/*  */}
      <Box sx={{ width: "100%", marginTop: "15px" }}>
        <FormControl required fullWidth>
          <InputLabel id="k-select-label">Top-K Result</InputLabel>
          <Select
            labelId="k-select-label"
            id="k-select"
            value={value_k}
            label="Top-K Result"
            onChange={handleValueKChange}
          >
            <MenuItem value={10}>10</MenuItem>
            <MenuItem value={20}>20</MenuItem>
            <MenuItem value={30}>30</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box sx={{ width: "100%", marginTop: "15px" }}>
        <FormControl required fullWidth>
          <InputLabel id="k-select-label">Window Frames</InputLabel>
          <Select
            labelId="k-select-label"
            id="k-select"
            value={valueFrame}
            label="Top-K Result"
            onChange={handleValueFrameChange}
          >
            <MenuItem value={10}>10</MenuItem>
            <MenuItem value={20}>20</MenuItem>
            <MenuItem value={30}>30</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          width: "100%",
          marginTop: "10px",
        }}
      >
        <Switch defaultChecked />
        <Box
          sx={{
            display: "flex",
            gap: "5px",
          }}
        >
          <Tooltip
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: "5px",
              border: "1px solid #ddd",
              width: "fit-content",
              padding: "3px",
              borderRadius: "5px",
              backgroundColor: "#f5f5f5",
            }}
            title="Refresh"
          >
            <IconButton>
              <ReplayIcon
                sx={{
                  fontSize: "16px",
                  fontWeight: "bold !important",
                  padding: "3px !important",
                }}
              />
            </IconButton>
          </Tooltip>

          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              gap: "5px",
              border: "1px solid #ddd",
              width: "50px",
              padding: "3px",
              borderRadius: "5px",
            }}
          >
            <BorderColorIcon
              sx={{
                fontSize: "16px",
                fontWeight: "bold",
                padding: "3px !important",
              }}
            />
            <Typography>0</Typography>
          </Box>
        </Box>
      </Box>
      <Box sx={{ width: "100%", marginTop: "20px" }}>
        <Button sx={{ width: "100%" }} variant="contained" endIcon={<SearchIcon />}>
          Send
        </Button>
      </Box>
    </Box>
  );
}
