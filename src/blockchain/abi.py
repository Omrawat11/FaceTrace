"""ABI definition for the EvidenceRegistry Ethereum smart contract."""

EVIDENCE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "fingerprint", "type": "bytes32"},
            {"internalType": "string", "name": "sourceUrl", "type": "string"},
        ],
        "name": "recordEvidence",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "fingerprint", "type": "bytes32"}
        ],
        "name": "getEvidence",
        "outputs": [
            {"internalType": "bool", "name": "exists", "type": "bool"},
            {"internalType": "address", "name": "recorder", "type": "address"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "blockNumber", "type": "uint256"},
            {"internalType": "string", "name": "sourceUrl", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "fingerprint", "type": "bytes32"}
        ],
        "name": "hasEvidence",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalRecords",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "fingerprint",
                "type": "bytes32",
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "recorder",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256",
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "blockNumber",
                "type": "uint256",
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "sourceUrl",
                "type": "string",
            },
        ],
        "name": "EvidenceRecorded",
        "type": "event",
    },
]
