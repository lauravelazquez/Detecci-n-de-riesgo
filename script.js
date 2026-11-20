
const form = document.getElementById("riskForm");

const latInput = document.getElementById("latitud");
const lonInput = document.getElementById("longitud");

const message = document.getElementById("message");

const btn = document.getElementById("evaluateBtn");



const map = L.map("map").setView(
    [19.4326, -99.1332],
    10
);


L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
    }
).addTo(map);


let marker;



function valid(lat, lon) {

    return (
        Number.isFinite(lat) &&
        Number.isFinite(lon) &&
        lat >= -90 &&
        lat <= 90 &&
        lon >= -180 &&
        lon <= 180
    );
}




function point(lat, lon, zoom = 16) {

    if (marker) {

        marker.setLatLng(
            [lat, lon]
        );

    } else {

        marker = L.marker(
            [lat, lon],
            {
                draggable: true
            }
        ).addTo(map);
    }


    marker
        .bindPopup(
            `Ubicación: ${lat.toFixed(6)}, ${lon.toFixed(6)}`
        )
        .openPopup();


    map.setView(
        [lat, lon],
        zoom
    );


    marker.off(
        "dragend"
    );


    marker.on(
        "dragend",
        (e) => {

            const p =
                e.target.getLatLng();

            latInput.value =
                p.lat.toFixed(6);

            lonInput.value =
                p.lng.toFixed(6);
        }
    );
}



map.on(
    "click",
    (e) => {

        latInput.value =
            e.latlng.lat.toFixed(6);

        lonInput.value =
            e.latlng.lng.toFixed(6);


        point(
            e.latlng.lat,
            e.latlng.lng
        );
    }
);



document
    .getElementById("locationBtn")
    .addEventListener(
        "click",
        () => {

            if (!navigator.geolocation) {

                setMessage(
                    "El navegador no permite obtener la ubicación.",
                    true
                );

                return;
            }


            setMessage(
                "Obteniendo ubicación..."
            );


            navigator.geolocation.getCurrentPosition(

                (posicion) => {

                    const lat =
                        posicion.coords.latitude;

                    const lon =
                        posicion.coords.longitude;


                    latInput.value =
                        lat.toFixed(6);

                    lonInput.value =
                        lon.toFixed(6);


                    point(
                        lat,
                        lon
                    );


                    setMessage(
                        "Ubicación cargada. Presiona Evaluar ubicación."
                    );
                },


                () => {

                    setMessage(
                        "No se pudo obtener la ubicación. Revisa los permisos del navegador.",
                        true
                    );
                },


                {
                    enableHighAccuracy: true,
                    timeout: 10000
                }
            );
        }
    );




function setMessage(
    text,
    error = false
) {

    message.textContent = text;

    message.className =
        "message" +
        (error ? " error" : "");
}




form.addEventListener(
    "submit",
    async (e) => {

        e.preventDefault();


        const lat =
            Number(
                latInput.value
            );

        const lon =
            Number(
                lonInput.value
            );


        // Validacion

        if (!valid(lat, lon)) {

            setMessage(
                "Ingresa una latitud entre -90 y 90 y una longitud entre -180 y 180.",
                true
            );

            return;
        }


        // Mostrar punto

        point(
            lat,
            lon
        );


        // Estado del boton

        btn.disabled = true;

        btn.textContent =
            "Analizando...";


        setMessage(
            "El modelo está evaluando la ubicación y buscando eventos históricos."
        );


        try {


            const respuesta = await fetch(
                `${window.API_URL}/predecir`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        latitud: lat,
                        longitud: lon
                    })
                }
            );


            const data =
                await respuesta.json();


            if (!respuesta.ok) {

                throw new Error(
                    data.error ||
                    "No fue posible evaluar la ubicación."
                );
            }



            showResult(
                data
            );


            setMessage(
                "Evaluación completada."
            );


        } catch (error) {

            setMessage(
                `${error.message} Verifica que el backend esté activo y que config.js tenga la URL correcta.`,
                true
            );


        } finally {

            btn.disabled = false;

            btn.textContent =
                "Evaluar ubicación";
        }
    }
);



function showResult(d) {


    const risk =
        d.riesgo_detectado ||
        "SIN RESULTADO";


    let cls =
        "revision";


    if (risk.includes("ALTO")) {

        cls = "alto";

    } else if (
        risk.includes("MEDIO")
    ) {

        cls = "medio";

    } else if (
        risk.includes("BAJO")
    ) {

        cls = "bajo";
    }



    document
        .getElementById("resultCard")
        .classList
        .remove("hidden");


    document
        .getElementById("riskValue")
        .textContent =
        risk;



    const badge =
        document.getElementById(
            "riskBadge"
        );


    badge.textContent =
        cls.toUpperCase();


    const colors = {

        alto: [
            "#fee2e2",
            "#991b1b"
        ],

        medio: [
            "#fef3c7",
            "#92400e"
        ],

        bajo: [
            "#dcfce7",
            "#166534"
        ],

        revision: [
            "#e2e8f0",
            "#334155"
        ]
    };


    badge.style.background =
        colors[cls][0];

    badge.style.color =
        colors[cls][1];



    const confianza =
        Number(
            d.confianza_porcentaje || 0
        );


    document
        .getElementById(
            "confidenceValue"
        )
        .textContent =
        `${confianza.toFixed(2)}%`;



    if (
        Array.isArray(
            d.coordenadas
        )
    ) {

        document
            .getElementById(
                "coordinatesValue"
            )
            .textContent =
            `${d.coordenadas[0]}, ${d.coordenadas[1]}`;
    }



    const eventos =
        d.eventos_historicos || {};


    // 250 metros

    const e250 =
        document.getElementById(
            "eventos250"
        );

    if (e250) {

        e250.textContent =
            eventos["250m"] ?? 0;
    }


    // 500 metros

    const e500 =
        document.getElementById(
            "eventos500"
        );

    if (e500) {

        e500.textContent =
            eventos["500m"] ?? 0;
    }


    // 1 kilometro

    const e1km =
        document.getElementById(
            "eventos1km"
        );

    if (e1km) {

        e1km.textContent =
            eventos["1km"] ?? 0;
    }


    // 2 kilometros

    const e2km =
        document.getElementById(
            "eventos2km"
        );

    if (e2km) {

        e2km.textContent =
            eventos["2km"] ?? 0;
    }



    const probabilidades =
        d.desglose_probabilidades ||
        {};


    const clases = [
        ["Alto", "ALTO"],
        ["Medio", "MEDIO"],
        ["Bajo", "BAJO"]
    ];


    clases.forEach(
        ([id, key]) => {

            const valor =
                Number(
                    probabilidades[key] || 0
                );


            const porcentaje =
                document.getElementById(
                    `prob${id}`
                );


            const barra =
                document.getElementById(
                    `bar${id}`
                );


            if (porcentaje) {

                porcentaje.textContent =
                    `${valor.toFixed(2)}%`;
            }


            if (barra) {

                barra.style.width =
                    `${
                        Math.min(
                            100,
                            Math.max(
                                0,
                                valor
                            )
                        )
                    }%`;
            }
        }
    );

    const protocolo =
        document.getElementById(
            "protocolValue"
        );


    if (protocolo) {

        protocolo.textContent =
            d.protocolo ||
            "Sin protocolo disponible.";
    }



    document
        .getElementById(
            "resultCard"
        )
        .scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
}
