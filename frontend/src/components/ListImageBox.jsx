import { Box, Skeleton, Grid } from "@mui/material";
import { useState } from "react";

export default function ListImageBox() {

    const [images, setImages] = useState([
        { id: 1, placeholder: true },
        { id: 2, placeholder: true },
        { id: 3, placeholder: true },
        { id: 4, placeholder: true },
        { id: 5, placeholder: true },
        { id: 6, placeholder: true },
        { id: 7, placeholder: true },
        { id: 8, placeholder: true },
        { id: 9, placeholder: true },
        { id: 10, placeholder: true },
        { id: 11, placeholder: true },
        { id: 12, placeholder: true },
    ]);

  return (
    <Box
      sx={{
        backgroundColor: "white",
        borderRadius: "8px !important",
        padding: "10px !important",
        height: "calc(100vh - 120px)",
        overflow: "auto"
      }}
    >
      <Grid container spacing={2}>
        {images.map((image) => (
          <Grid item size={{ xs: 12, sm: 6, md: 4, lg: 3 }} key={image.id}>
            <Box
              sx={{
                cursor: "pointer",
                transition: "transform 0.2s, box-shadow 0.2s",
                "&:hover": {
                  transform: "scale(1.02)",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.15)"
                }
              }}
            >
              {image.placeholder ? (
                <Skeleton 
                  variant="rounded" 
                  width="100%" 
                  height={200}
                  sx={{
                    borderRadius: "8px"
                  }}
                />
              ) : (
                <img 
                  src={image.src} 
                  alt={image.alt}
                  style={{
                    width: "100%",
                    height: "200px",
                    objectFit: "cover",
                    borderRadius: "8px"
                  }}
                />
              )}
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
