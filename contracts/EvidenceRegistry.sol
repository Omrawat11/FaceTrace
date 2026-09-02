// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title EvidenceRegistry
 * @dev Minimal on-chain registry for cryptographic evidence fingerprints.
 * 
 * IMPORTANT DESIGN PRINCIPLES:
 * 1. AI performs biometric face similarity matching off-chain.
 * 2. The smart contract NEVER stores raw biometric data or private images.
 * 3. Only the 32-byte deterministic SHA-256 fingerprint of the canonical evidence
 *    bundle is anchored on-chain for tamper-evident verification.
 * 4. Blockchain verifies evidence integrity and temporal existence; it does NOT
 *    claim to "prove someone's identity".
 */
contract EvidenceRegistry {

    struct EvidenceRecord {
        bytes32 fingerprint;
        address recorder;
        uint256 timestamp;
        uint256 blockNumber;
        string sourceUrl;
    }

    // Mapping from cryptographic SHA-256 fingerprint (bytes32) to evidence record
    mapping(bytes32 => EvidenceRecord) private _records;

    // Total count of uniquely recorded evidence fingerprints
    uint256 public totalRecords;

    // Events
    event EvidenceRecorded(
        bytes32 indexed fingerprint,
        address indexed recorder,
        uint256 timestamp,
        uint256 blockNumber,
        string sourceUrl
    );

    // Custom errors for gas efficiency
    error EvidenceAlreadyRecorded(bytes32 fingerprint, uint256 recordedAt);
    error InvalidFingerprint();
    error EmptySourceUrl();

    /**
     * @notice Records a new cryptographic evidence fingerprint onto the blockchain.
     * @param fingerprint The 32-byte SHA-256 hash of the canonical evidence object.
     * @param sourceUrl The public URL where the candidate image was legitimately discovered.
     * @return success True if the recording succeeded.
     */
    function recordEvidence(bytes32 fingerprint, string calldata sourceUrl) external returns (bool) {
        if (fingerprint == bytes32(0)) {
            revert InvalidFingerprint();
        }
        if (bytes(sourceUrl).length == 0) {
            revert EmptySourceUrl();
        }
        if (_records[fingerprint].timestamp != 0) {
            revert EvidenceAlreadyRecorded(fingerprint, _records[fingerprint].timestamp);
        }

        _records[fingerprint] = EvidenceRecord({
            fingerprint: fingerprint,
            recorder: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number,
            sourceUrl: sourceUrl
        });

        totalRecords += 1;

        emit EvidenceRecorded(
            fingerprint,
            msg.sender,
            block.timestamp,
            block.number,
            sourceUrl
        );

        return true;
    }

    /**
     * @notice Checks if a fingerprint has been anchored on-chain.
     * @param fingerprint The 32-byte SHA-256 evidence fingerprint.
     * @return exists True if the fingerprint exists in the registry.
     */
    function hasEvidence(bytes32 fingerprint) external view returns (bool) {
        return _records[fingerprint].timestamp != 0;
    }

    /**
     * @notice Retrieves full evidence details for a given fingerprint.
     * @param fingerprint The 32-byte SHA-256 evidence fingerprint.
     * @return exists Whether the record exists.
     * @return recorder Address that submitted the transaction.
     * @return timestamp Block timestamp when recorded.
     * @return blockNumber Block number in which it was mined.
     * @return sourceUrl Public source URL associated with the evidence.
     */
    function getEvidence(bytes32 fingerprint) external view returns (
        bool exists,
        address recorder,
        uint256 timestamp,
        uint256 blockNumber,
        string memory sourceUrl
    ) {
        EvidenceRecord storage record = _records[fingerprint];
        if (record.timestamp == 0) {
            return (false, address(0), 0, 0, "");
        }
        return (
            true,
            record.recorder,
            record.timestamp,
            record.blockNumber,
            record.sourceUrl
        );
    }
}
