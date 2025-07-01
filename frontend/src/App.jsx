import {
  Box,
  Chip,
  Container,
  FormControl,
  MenuItem,
  Select,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import SettingsIcon from "@mui/icons-material/Settings";
import CachedIcon from "@mui/icons-material/Cached";
import AirplayIcon from "@mui/icons-material/Airplay";
import ContentPasteIcon from "@mui/icons-material/ContentPaste";
import QueryBox from "./components/QueryBox";
import ListImageBox from "./components/ListImageBox";
import { useState } from "react";

function App() {
  // change state
  const [state, setState] = useState("state_1");
  const handleChange = (event, newState) => {
    if (newState !== null) {
      setState(newState);
    }
  };

  //
  const [option, setOption] = useState("KSI");
  const handleOptionChange = (event) => {
    setOption(event.target.value);
  };

  const [option2, setOption2] = useState("KSI");
  const handleOptionChange2 = (event) => {
    setOption2(event.target.value);
  };

  return (
    <Container
      sx={{
        maxWidth: "100% !important",
        margin: "0 !important",
        backgroundColor: "#dfe6e9 !important",
        padding: "10px !important",
        height: "100vh !important",
      }}
    >
      <Box
        className="header"
        sx={{
          boxSizing: "border-box",
          display: "flex",
          justifyContent: "space-evenly",
          alignItems: "center",
          borderRadius: "8px !important",
          backgroundColor: "white !important",
          paddingLeft: "100px !important",
        }}
      >
        <Box>
          <ToggleButtonGroup
            color="primary"
            value={state}
            exclusive
            onChange={handleChange}
            sx={{ backgroundColor: "grey.100" }}
          >
            <ToggleButton
              value="state_1"
              sx={{
                textTransform: "none !important",
                backgroundColor:
                  state == "state_1"
                    ? "white !important"
                    : "grey.100 !important",
              }}
            >
              <Typography
                sx={{
                  fontSize: "15px",
                }}
              >
                Stage 1
              </Typography>
            </ToggleButton>
            <ToggleButton
              value="state_2"
              sx={{
                textTransform: "none !important",
                backgroundColor:
                  state == "state_2"
                    ? "white !important"
                    : "grey.100 !important",
              }}
            >
              <Typography
                sx={{
                  fontSize: "15px",
                }}
              >
                Stage 2
              </Typography>
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
        <Box
          sx={{
            display: "flex",
            gap: "5px",
          }}
        >
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              borderRadius: "50% !important",
              backgroundColor: "grey.200",
              width: "30px",
              height: "30px",
              padding: "5px",
            }}
          >
            <SettingsIcon />
          </Box>
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              borderRadius: "50% !important",
              backgroundColor: "#f39c12",
              width: "30px",
              height: "30px",
              padding: "5px",
            }}
          >
            <CachedIcon />
          </Box>
        </Box>
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: "5px",
          }}
        >
          <FormControl sx={{ m: 1, minWidth: 100 }} size="small">
            <Select
              value={option}
              displayEmpty
              inputProps={{ "aria-label": "Without label" }}
              onChange={handleOptionChange}
            >
              <MenuItem value={"KSI"}>KSI</MenuItem>
              <MenuItem value={"KSI_2"}>KSI_2</MenuItem>
              <MenuItem value={"KSI_3"}>KSI_3</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ m: 1, minWidth: 100 }} size="small">
            <Select
              value={option2}
              displayEmpty
              inputProps={{ "aria-label": "Without label" }}
              onChange={handleOptionChange2}
            >
              <MenuItem value={"KSI"}>KSI</MenuItem>
              <MenuItem value={"KSI_2"}>KSI_2</MenuItem>
              <MenuItem value={"KSI_3"}>KSI_3</MenuItem>
            </Select>
          </FormControl>

          <Chip
            sx={{
              backgroundColor: "#fab1a0",
            }}
            icon={
              <AirplayIcon
                sx={{
                  width: "15px !important",
                }}
              />
            }
            label="2"
          />

          <Chip
            sx={{
              backgroundColor: "#ffeaa7",
            }}
            icon={
              <ContentPasteIcon
                sx={{
                  width: "15px !important",
                }}
              />
            }
            label="2"
          />
        </Box>
      </Box>

      {/* body */}
      <Box sx={{
        display: "flex",
      }}>
        {/* left panel */}
        <Box sx={{
          width: "20% !important",
          marginTop: "10px !important",
          paddingRight: "10px !important",
          height: "100% !important",
        }}>
            <QueryBox />
        </Box>
        {/* center panel */}
        <Box sx = {{
          width: "80% !important",
          marginTop: "10px !important",
        }}>
          <ListImageBox />
        </Box>
      </Box>
    </Container>
  );
}

export default App;
